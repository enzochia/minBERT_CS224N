# CS 224N Final Project 2024 - Multitask BERT
[link to project poster](Task_specific_attention.pdf)

[link to project report](Task_specific_attention_report.pdf)

## Abstract
In this project I propose and explore task-specific attention mechanism, which takes BERT foundation model output and is intentionally overfitted for each task. Experiments proved that this mechanism can help to improve model performance when finetune data is relatively large. Corresponding training cost is not significantly higher compared to traditional finetunes. In this exploration, task-specific attention layers improve paraphrase detection accuracy from 0.551 to 0.842, with a 10% increase of finetune training time.

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
