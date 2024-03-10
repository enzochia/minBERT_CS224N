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
import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel, BertSelfAttention
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

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask
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
        self.self_attn = BertSelfAttention(self.bert.config)
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


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''

        # TODO: attention
        # TODO: another dropout
        sent_encode = self.forward(input_ids, attention_mask)
        # proj = self.sentiment_proj(sent_encode)
        # pred = F.softmax(proj, dim=-1)

        # attention
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.bert.dtype)
        attn_seq = self.self_attn(sent_encode, extended_attention_mask)
        # attn = attn_seq[:, 0]
        attn = self.get_mean_bert_output(attn_seq, attention_mask, True)
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
        # sent_encode_1 = self.forward(input_ids_1, attention_mask_1)
        # sent_encode_2 = self.forward(input_ids_2, attention_mask_2)
        sent_encode_1 = self.forward(input_ids_1, attention_mask_1)
        sent_encode_1 = self.get_mean_bert_output(sent_encode_1, attention_mask_1, True)
        sent_encode_1 = self.dropout(sent_encode_1)
        sent_encode_2 = self.forward(input_ids_2, attention_mask_2)
        sent_encode_2 = self.get_mean_bert_output(sent_encode_2, attention_mask_2, True)
        sent_encode_2 = self.dropout(sent_encode_2)
        proj_1 = self.similarity_proj(sent_encode_1)
        proj_2 = self.similarity_proj(sent_encode_2)
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
        #     b_ids_1 = b_ids_1.to(device)
        #     b_mask_1 = b_mask_1.to(device)
        #     b_ids_2 = b_ids_2.to(device)
        #     b_mask_2 = b_mask_2.to(device)
        #     b_labels = b_labels.float().to(device)
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
