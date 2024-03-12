conda activate cs224n_dfp
python3 multitask_classifier.py --option pretrain --lr 1e-3
python3 multitask_classifier.py --option finetune --lr 1e-5
