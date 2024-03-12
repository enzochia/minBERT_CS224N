'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import time
import sys
import math
import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel, BertSelfAttention, BertLayer
from optimizer import AdamW
from tqdm import tqdm
from utils import *

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask, align_pair_sents
eps = 1e-7

TQDM_DISABLE=False

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
NUM_HIDDEN_LAYERS_SST = 8
NUM_HIDDEN_LAYERS_STS = 1

class BertCrossAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    nn.init.uniform_(self.query.weight, 0.9, 1)
    nn.init.uniform_(self.key.weight, 0.9, 1)
    nn.init.uniform_(self.value.weight, 0.9, 1)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()
    self.resid_drop = nn.Dropout(0.1)
    self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    bs, num_heads, seq_len, head_size = query.size()
    # (bs, num_heads, seq_len, head_size) x (bs, num_heads, head_size, seq_len) ->
    # (bs, num_heads, seq_len, seq_len)
    att = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
    # pay attention here, this may be implemented wrong
    # https://huggingface.co/docs/transformers/glossary#attention-mask
    att = att.masked_fill(attention_mask[:bs, :, :, :seq_len] < 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.dropout(att)
    # (bs, num_heads, seq_len, seq_len) x (bs, num_heads, seq_len, head_size)
    # -> (bs, num_heads, seq_len, head_size)
    y = torch.matmul(att, value)
    y = y.transpose(1, 2).contiguous().view(bs, seq_len, num_heads * head_size)
    y = self.resid_drop(self.output_proj(y))
    return (y)


  def forward(self, hidden_states_kv, hidden_states_q, attention_mask_kv):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    key_layer = self.transform(hidden_states_kv, self.key)
    value_layer = self.transform(hidden_states_kv, self.value)
    query_layer = self.transform(hidden_states_q, self.query)
    seq_attn = self.attention(key_layer, query_layer, value_layer, attention_mask_kv)
    # first_tk = seq_attn[:, 0]
    # first_tk = self.pooler_dense(first_tk)
    # first_tk = self.pooler_af(first_tk)
    # return {"seq_attn": seq_attn,
    #         "CLS_attn": first_tk}
    return(seq_attn)

class BertCrossAttnLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    # self.self_attention = BertSelfAttention(config)
    self.cross_attention = BertCrossAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied
    ln_layer: the layer norm to be applied
    """
    output = dropout(dense_layer(output))
    output = ln_layer(input + output)
    return(output)


  def forward(self, hidden_states_1, hidden_states_2,
              attention_mask_1, attention_mask_2):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    # attn_value = self.self_attention(hidden_states, attention_mask)
    # sent_encode_i = self.cross_attn(hidden_states_i, hidden_states_ii, attention_mask_i)
    # sent_encode_ii = self.cross_attn(hidden_states_ii, hidden_states_i, attention_mask_ii)
    attn_value_1 = self.cross_attention(hidden_states_1, hidden_states_2, attention_mask_1)
    attn_value_2 = self.cross_attention(hidden_states_2, hidden_states_1, attention_mask_2)
    attn_value_1 = self.add_norm(hidden_states_1, attn_value_1, self.attention_dense,
                                 self.attention_dropout, self.attention_layer_norm)
    attn_value_2 = self.add_norm(hidden_states_2, attn_value_2, self.attention_dense,
                                 self.attention_dropout, self.attention_layer_norm)
    output_1 = self.interm_dense(attn_value_1)
    output_1 = self.interm_af(output_1)
    output_1 = self.add_norm(attn_value_1, output_1, self.out_dense,
                             self.out_dropout, self.out_layer_norm)
    output_2 = self.interm_dense(attn_value_2)
    output_2 = self.interm_af(output_2)
    output_2 = self.add_norm(attn_value_2, output_2, self.out_dense,
                             self.out_dropout, self.out_layer_norm)
    return(output_1, output_2)

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.seq_length = self.bert.config.max_position_embeddings
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        self.sentiment_proj = nn.Linear(config.hidden_size, len(config.num_labels))
        self.paraphrase_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.similarity_proj = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.self_attn = BertSelfAttention(self.bert.config)
        self.bert_layers_sst = nn.ModuleList([BertLayer(self.bert.config)
                                              for _ in range(NUM_HIDDEN_LAYERS_SST)])
        self.bert_layers_sts = nn.ModuleList([BertCrossAttnLayer(self.bert.config)
                                              for _ in range(NUM_HIDDEN_LAYERS_STS)])
        self.cross_attn = BertCrossAttention(self.bert.config)
        # self.agg_proj = nn.Linear(self.seq_length, 1)
        # ### TODO
        # raise NotImplementedError

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        """
        performance:
                                                          dev sentiment acc   dev paraphrase acc  dev sts corr  training time (secs)
        order: sst, para, sts                   pretrain            0.457                0.381         0.632        14673
                                                finetune            0.453                0.543         0.611        37164
        order: sst, para, sts (mse loss)        pretrain            0.461                0.388         0.030        13800
                                                finetune            0.268                0.507         0.290        35855
        order: sst, sts, para                   pretrain            0.467                0.379         0.598        14637
                                                finetune            0.481                0.488         0.171        37126
        order: para, sst, sts                   pretrain            0.459                0.384         0.621        13932
                                                finetune            0.262                0.460         0.298        36022
        order: para, sts, sst                   pretrain            0.451                0.380         0.636        14500
                                                finetune            0.500                0.485         0.344        37124
        order: sts, sst, para                   pretrain            0.473                0.390         0.563        14627
                                                finetune            0.361                0.568         0.180        37118
        order: sts, sst, para                   pretrain            0.463                0.386         0.594        13916
                                                finetune            0.369                0.407         0.193        35927
        task specific finetune                  pretrain            0.389                0.401         0.298        868 + 13274 + 864
        first token                             finetune            0.504                0.595         0.587        1886 + 33475 + 1822
        task specific finetune                  pretrain            0.460                0.388         0.643        819 + 13259 + 837
        mean of seq                             finetune            0.515                0.550         0.854        1829 + 33475 + 1715
                                                
        
        Note: 20240307: order: sst, sts, para. on vm ii  -- 03/07 am
                        order: para, sst, sts. on vm iii -- 03/07 am
                        order: para, sts, sst. on vm iv  -- 03/07 am
                        order: sts, sst, para. on vm ii  -- 03/07 pm
                        order: sts, para, sst. on vm iii -- 03/07 pm
                        
                        loss: mse for sts   
                        order: sst, para, sts. on vm v   -- 03/07 am
                        
                        single finetune
                        sst.                   on vm v   -- 03/07 pm
                        sts 2x.                on vm v   -- 03/07 pm
                        sts 4x.                on vm v   -- 03/07 pm
                        sts 8x.                on vm v   -- 03/07 pm
                        para 2x.               on vm v   -- 03/07 pm
                        para 1x.               on vm ii  -- 03/08 am
                        para 4x.               on vm iii -- 03/08 am
                        para 8x.               on vm iv  -- 03/08 am
                        sts 1x.                on vm v   -- 03/08 am
                        
                        pooler:
                        sst.                   on vm ii  -- 03/09 am
                        para.                  on vm iii -- 03/09 am
                        sts.                   on vm iv  -- 03/09 am
                        
                        attention:
                        sst (pooler).          on vm ii  -- 03/09 am
                        sst (seq).             on vm iv  -- 03/09 pm
                        sst (1 BertLayer, no 
                             final dropout)    on vm ii  -- 03/10 pm
                        sst (2 BertLayer, no 
                             final dropout)    on vm iii -- 03/10 pm      
                        sst (4 BertLayer, no 
                             final dropout)    on vm iv  -- 03/10 pm                  
                        sst (8 BertLayer, no 
                             final dropout)    on vm v   -- 03/10 pm      
                        sts (1 BertCrossAttnLayer, no 
                             final dropout)    on vm ii  -- 03/10 pm    
                        sts (2 BertCrossAttnLayer, no 
                             final dropout)    on vm iii -- 03/10 pm   
                        sst (1 BertLayer, uniform init, no 
                             final dropout)    on vm ii  -- 03/11 pm
                        sst (2 BertLayer, uniform init, no 
                             final dropout)    on vm iii -- 03/11 pm    
                        sst (4 BertLayer, uniform init, no 
                             final dropout)    on vm iv  -- 03/11 pm
                        sst (8 BertLayer, uniform init, no 
                             final dropout)    on vm v   -- 03/11 pm   
                        sst (1 BertLayer, uniform init,  
                             final dropout)    on vm ii  -- 03/11 pm
                        sst (2 BertLayer, uniform init,  
                             final dropout)    on vm iii -- 03/11 pm   
                        sst (4 BertLayer, uniform init,  
                             final dropout)    on vm iv  -- 03/11 pm
                        sst (8 BertLayer, uniform init,  
                             final dropout)    on vm v   -- 03/11 pm                                
        """
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        # # return first token's BERT attention
        # encode_dict = self.bert(input_ids, attention_mask)
        # pooler_output = encode_dict['pooler_output']
        # pooler_output = self.dropout(pooler_output)

        # # return mean of sequences' BERT attention
        # encode_dict = self.bert(input_ids, attention_mask)
        # seq_hidden = encode_dict['last_hidden_state']
        # pooler_output = self.get_mean_bert_output(seq_hidden, attention_mask, True)
        # pooler_output = self.dropout(pooler_output)

        # return sequence for task-specific attention blocks
        encode_dict = self.bert(input_ids, attention_mask)
        seq_hidden = encode_dict['last_hidden_state']
        pooler_output = seq_hidden

        return(pooler_output)

    def get_mean_bert_output(self, seq_encode, attention_mask, mask_excluded=True):
        """"
        Given a sequence's all token's BERT attention outputs, return mean attention,
        with or without masks' attention outputs.
        """
        if mask_excluded:
            mask_3d = attention_mask[:, :, None]
            mask_sum = mask_3d.sum(dim=-2)
            mat_product = seq_encode * mask_3d
            repsentation = mat_product.sum(dim=-2) / mask_sum
        else:
            repsentation = seq_encode.sum(dim=-2) / seq_encode.size()[-2]
        return(repsentation)

    def get_bert_cross_attn(self, hidden_states_i, hidden_states_ii,
                            attention_mask_i, attention_mask_ii, first_tk = True):
        sent_encode_i = self.cross_attn(hidden_states_i, hidden_states_ii, attention_mask_i)
        sent_encode_ii = self.cross_attn(hidden_states_ii, hidden_states_i, attention_mask_ii)
        if first_tk:
            # tk_i = sent_encode_i['CLS_attn']
            # tk_ii = sent_encode_ii['CLS_attn']
            tk_i = sent_encode_i[:, 0]
            tk_ii = sent_encode_ii[:, 0]
        else:
            tk_i = self.get_mean_bert_output(sent_encode_i, attention_mask_ii, True)
            tk_ii = self.get_mean_bert_output(sent_encode_ii, attention_mask_i, True)
        return(tk_i, tk_ii)

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''

        # TODO: attention
        # TODO: another dropout

        # # direct project without attentions
        # sent_encode = self.forward(input_ids, attention_mask)
        # proj = self.sentiment_proj(sent_encode)
        # pred = F.softmax(proj, dim=-1)

        # project using attentions
        sent_encode = self.forward(input_ids, attention_mask)
        # sent_encode = self.dropout(sent_encode)
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.bert.dtype)
        # Pass the hidden states through the encoder layers.
        for i, layer_module in enumerate(self.bert_layers_sst):
            # Feed the encoding from the last bert_layer to the next.
            sent_encode = layer_module(sent_encode, extended_attention_mask)
        attn = sent_encode[:, 0]
        # attn = self.get_mean_bert_output(attn_seq, attention_mask, True)
        attn = self.dropout(attn)
        proj = self.sentiment_proj(attn)
        pred = F.softmax(proj, dim=-1)
        return (pred)


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''

        # TODO: attention
        # TODO: CNN
        # TODO: another dropout
        # TODO: small layer(s)
        # TODO: use less layers
        # sent_encode_1 = self.forward(input_ids_1, attention_mask_1)
        # sent_encode_2 = self.forward(input_ids_2, attention_mask_2)
        sent_encode_1 = self.forward(input_ids_1, attention_mask_1)
        sent_encode_1 = self.get_mean_bert_output(sent_encode_1, attention_mask_1, True)
        sent_encode_1 = self.dropout(sent_encode_1)
        sent_encode_2 = self.forward(input_ids_2, attention_mask_2)
        sent_encode_2 = self.get_mean_bert_output(sent_encode_2, attention_mask_2, True)
        sent_encode_2 = self.dropout(sent_encode_2)
        proj_1 = self.paraphrase_proj(sent_encode_1)
        proj_2 = self.paraphrase_proj(sent_encode_2)
        product = proj_1 * proj_2
        pred = product.sum(dim=1)
        return(pred)


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''

        # TODO: attention
        # TODO: another dropout
        # TODO: bigger layer(s)

        # # direct project without attentions
        # # sent_encode_1 = self.forward(input_ids_1, attention_mask_1)
        # # sent_encode_2 = self.forward(input_ids_2, attention_mask_2)
        # sent_encode_1 = self.forward(input_ids_1, attention_mask_1)
        # sent_encode_1 = self.get_mean_bert_output(sent_encode_1, attention_mask_1, True)
        # sent_encode_1 = self.dropout(sent_encode_1)
        # sent_encode_2 = self.forward(input_ids_2, attention_mask_2)
        # sent_encode_2 = self.get_mean_bert_output(sent_encode_2, attention_mask_2, True)
        # sent_encode_2 = self.dropout(sent_encode_2)
        # proj_1 = self.similarity_proj(sent_encode_1)
        # proj_2 = self.similarity_proj(sent_encode_2)
        # product = proj_1 * proj_2
        # pred = product.sum(dim=1)

        # project using attentions
        extended_attention_mask_1: torch.Tensor = get_extended_attention_mask(attention_mask_1, self.bert.dtype)
        extended_attention_mask_2: torch.Tensor = get_extended_attention_mask(attention_mask_2, self.bert.dtype)
        sent_encode_1 = self.forward(input_ids_1, attention_mask_1)
        sent_encode_2 = self.forward(input_ids_2, attention_mask_2)
        # sent_encode_1 = self.dropout(sent_encode_1)
        # sent_encode_2 = self.dropout(sent_encode_2)

        for i, layer_module in enumerate(self.bert_layers_sts):
            # Feed the encoding from the last bert_layer to the next.
            sent_encode_1, sent_encode_2 = layer_module(sent_encode_1, sent_encode_2,
                                                        extended_attention_mask_1, extended_attention_mask_2)
        tk_i, tk_ii = self.get_bert_cross_attn(sent_encode_1, sent_encode_2, extended_attention_mask_1,
                                               extended_attention_mask_2, True)
        tk_i = self.dropout(tk_i)
        tk_ii = self.dropout(tk_ii)
        # tk_i = self.dropout(tk_i)
        # tk_ii = self.dropout(tk_ii)
        proj_1 = self.similarity_proj(tk_i)
        proj_2 = self.similarity_proj(tk_ii)
        product = proj_1 * proj_2
        pred = product.sum(dim=1)
        return(pred)




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")



def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f'working on device: {device}')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    cosSim = nn.CosineSimilarity(dim=1, eps=1e-7)
    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_score = 0

    # Run for the specified number of epochs.
    # sst, para, and sts are finetuned separately.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-sst-epoch-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        # for batch in tqdm(para_train_dataloader, desc=f'train-para-epoch-{epoch}', disable=TQDM_DISABLE):
        #     b_ids_1, b_mask_1, \
        #     b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
        #                                    batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
        #     b_ids_1 = b_ids_1.to(device)
        #     b_mask_1 = b_mask_1.to(device)
        #     b_ids_2 = b_ids_2.to(device)
        #     b_mask_2 = b_mask_2.to(device)
        #     b_labels = b_labels.float().to(device)
        #
        #     optimizer.zero_grad()
        #     logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        #     loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1), reduction='sum') / args.batch_size
        #     # y_hat = logits.sigmoid().round()
        #     # loss = -torch.eq(y_hat, b_labels).float().mean()
        #     loss.backward()
        #     optimizer.step()
        #
        #     train_loss += loss.item()
        #     num_batches += 1

        # for batch in tqdm(sts_train_dataloader, desc=f'train-sts-epoch-{epoch}', disable=TQDM_DISABLE):
        #     b_ids_1, b_mask_1, \
        #     b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],
        #                                    batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
        #     b_ids_1, b_ids_2, b_mask_1, b_mask_2 = align_pair_sents(b_ids_1, b_ids_2, b_mask_1, b_mask_2)
        #     b_ids_1 = b_ids_1.int().to(device)
        #     b_mask_1 = b_mask_1.int().to(device)
        #     b_ids_2 = b_ids_2.int().to(device)
        #     b_mask_2 = b_mask_2.int().to(device)
        #     b_labels = b_labels.int().float().to(device)
        #
        #     optimizer.zero_grad()
        #     logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        #     x1 = logits.view(-1, args.batch_size)
        #     x2 = b_labels.view(-1, args.batch_size)
        #     # this is actually pearson correlation
        #     loss = -cosSim(x1 - x1.mean(dim=1, keepdim=True),
        #                    x2 - x2.mean(dim=1, keepdim=True)) / args.batch_size
        #     # logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2).sigmoid() * 5.0
        #     # loss = F.mse_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size
        #     loss.backward()
        #     optimizer.step()
        #
        #     train_loss += loss.item()
        #     num_batches += 1

        train_loss = train_loss / (num_batches)

        sst_train_acc, _, _, \
        para_train_acc, _, _, \
        sts_train_corr, _, _ =  model_eval_multitask(sst_train_dataloader,
                                                     para_train_dataloader,
                                                     sts_train_dataloader,
                                                     model, device)

        sst_dev_acc, _, _, \
        para_dev_acc, _, _, \
        sts_dev_corr, _, _ =  model_eval_multitask(sst_dev_dataloader,
                                                   para_dev_dataloader,
                                                   sts_dev_dataloader,
                                                   model, device)
        train_score = (sst_train_acc + para_train_acc + ((1 + sts_train_corr) / 2)) / 3
        dev_score = (sst_dev_acc + para_dev_acc + ((1 + sts_dev_corr) / 2)) / 3

        if dev_score > best_dev_score:
            best_dev_score = dev_score
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train score :: {train_score :.3f}, dev score :: {dev_score :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        print(f'working on device: {device}')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_start = time.time()
    old_stdout = sys.stdout
    log_file = open("message" + args.option + str(int(train_start)) +".log", "w")
    sys.stdout = log_file

    train_multitask(args)
    test_start = time.time()
    print(f'Training cost {int(test_start - train_start)} seconds')
    test_multitask(args)
    test_end = time.time()
    print(f'Testing cost {int(test_end - test_start)} seconds')

    sys.stdout = old_stdout
    log_file.close()
