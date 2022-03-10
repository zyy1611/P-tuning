import json
import os
import logging
import warnings

import transformers
import numpy as np
from sklearn import metrics
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForMaskedLM

warnings.filterwarnings("ignore")

classes = ['sportsman', 'singer', 'actor', 'politician', 'royal', 'cosmonaut', 'country', 'city', 'brand', 'airplane',
           'car', 'train', 'club', 'sports team', 'company', 'army', 'mall', 'school', 'hospital', 'airport', 'stadium',
           'government', 'body of water', 'mountain', 'park', 'island', 'movie', 'music', 'broadcast', 'video game',
           'war', 'disaster', 'competition', 'festival', 'language', 'award', 'disease']
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
transformers.set_seed(1)
logging.basicConfig(level=logging.INFO)
new_tokens = ['cosmonaut', 'sports team', 'video game', 'body of water']


class LecCallTag():
    def __int__(self):
        self.label_ids = None

    # 原始样本统计
    def data_show(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info("获取数据：%s" % len(data))
        tags_data_dict = {}
        for line in data:
            text_label = line.strip().split('    ')
            if text_label[0] in tags_data_dict:
                tags_data_dict[text_label[0]].append(text_label[1])
            else:
                tags_data_dict[text_label[0]] = [text_label[1]]
        logging.info("其中，各分类数量：")
        for k, v in tags_data_dict.items():
            logging.info("%s: %s" % (k, len(v)))
        return tags_data_dict

    # 数据处理
    def data_process(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [line.strip().split('    ') for line in json.load(f)]
        text = ['entity is [MASK],' + _[1] for _ in data]
        label = ['entity is ' + _[0] + ',' + _[1] for _ in data]
        return text, label

    # model, tokenizer
    def create_model_tokenizer(self, model_name, n_label=0):
        tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=True)
        model = BertForMaskedLM.from_pretrained(model_name)
        num_added_toks = tokenizer.add_tokens(new_tokens)  # 返回一个数，表示加入的新词数量，在这里是2

        # 关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.save_pretrained("raw_bert/bert-base-uncased")
        self.label_ids = np.array([tokenizer.encode(l)[1] for l in classes])
        return tokenizer, model

    # 构建dataset
    def create_dataset(self, text, label, tokenizer, max_len):
        X_train, X_test, Y_train, Y_test = train_test_split(text, label, test_size=0.2, random_state=1)
        logging.info('训练集：%s条，\n测试集：%s条' % (len(X_train), len(X_test)))
        train_dict = {'text': X_train, 'label_text': Y_train}
        test_dict = {'text': X_test, 'label_text': Y_test}
        train_dataset = Dataset.from_dict(train_dict)
        test_dataset = Dataset.from_dict(test_dict)

        def preprocess_function(examples):
            text_token = tokenizer(examples['text'], padding=True, truncation=True, max_length=max_len)
            text_token['labels'] = np.array(
                tokenizer(examples['label_text'], padding=True, truncation=True, max_length=max_len)[
                    "input_ids"])  # 注意数据类型
            return text_token

        train_dataset = train_dataset.map(preprocess_function, batched=True)
        test_dataset = test_dataset.map(preprocess_function, batched=True)
        return train_dataset, test_dataset

    # 构建trainer
    def create_trainer(self, model, train_dataset, test_dataset, checkpoint_dir, batch_size):
        args = TrainingArguments(
            checkpoint_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=15,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        def compute_metrics(pred):
            labels = pred.label_ids[:, 3]
            preds = pred.predictions[:, 3]
            y_pred = preds[:, self.label_ids]
            y_pred = y_pred.argmax(axis=1)
            res = np.array([self.label_ids[obj] for obj in y_pred])
            precision, recall, f1, _ = precision_recall_fscore_support(labels, res, average='weighted')
            acc = accuracy_score(labels, res)
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            # tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        return trainer


def main():
    lct = LecCallTag()
    data_file = './data/fr_en_v1/data.txt'
    checkpoint_dir = "./checkpoint_fr_en_v1_2/"
    batch_size = 128
    max_len = 150
    tags_data = lct.data_show(data_file)
    text, label = lct.data_process(data_file)
    tokenizer, model = lct.create_model_tokenizer("raw_bert/bert-base-uncased")

    train_dataset, test_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    trainer = lct.create_trainer(model, train_dataset, test_dataset, checkpoint_dir, batch_size)
    trainer.train()


if __name__ == '__main__':
    main()
