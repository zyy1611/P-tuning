import os
import torch
import logging
import datasets
import transformers
import numpy as np
import torch.nn as nn
from sklearn import metrics
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
transformers.set_seed(1)
logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prp_len = 2  # prompt token长度


# 通过LSTM寻找prompt的embedding
class MyModel(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.dim = 384
        self.emb = nn.Embedding(prp_len + 1, self.dim)
        self.bi_lstm = nn.LSTM(self.dim, self.dim, 2, bidirectional=True)
        self.b_emb = self.get_input_embeddings()
        self.line1 = nn.Linear(768, 768)
        self.line2 = nn.Linear(768, 768)
        self.line3 = nn.Linear(768, 768)
        self.relu = nn.ReLU()

    def forward(
            self,
            input_ids=None,  # [CLS] e(p) e(p) [MASK] e(input_ids)
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,  # [CLS] -100 -100 label e(input_ids)
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        p = self.emb(torch.LongTensor([range(1, prp_len + 1)] * input_ids.shape[0]).to(device))  # 若用GPU则要注意将数据导入cuda
        p = self.bi_lstm(p)[0]
        p = self.relu(self.line1(p))
        p = self.relu(self.line2(p))
        p = self.relu(self.line3(p))
        inputs_embeds = self.b_emb(input_ids)
        inputs_embeds[:, 1:prp_len + 1, :] = p

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LecCallTag():

    # 原始样本统计
    def data_show(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
        logging.info("获取数据：%s" % len(data))
        tags_data_dict = {}
        for line in data:
            text_label = line.strip().split('\t')
            if text_label[1] in tags_data_dict:
                tags_data_dict[text_label[1]].append(text_label[0])
            else:
                tags_data_dict[text_label[1]] = [text_label[0]]
        logging.info("其中，各分类数量：")
        for k, v in tags_data_dict.items():
            logging.info("%s: %s" % (k, len(v)))
        return tags_data_dict

    # 数据处理
    def data_process(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [line.strip().split('\t') for line in f.readlines()]
        self.lable2idx1 = {'天气好': '好', '天气良': '良', '天气差': '差', '其他': '无'}
        text = ['[MASK]' * (prp_len + 1) + _[0] for _ in data]
        label = [self.lable2idx1[_[1]] * (prp_len + 1) + _[0] for _ in data]
        return text, label

    # model, tokenizer
    def create_model_tokenizer(self, model_name, n_label=0):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = MyModel.from_pretrained(model_name)
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
                tokenizer(examples['label_text'], padding=True, truncation=True, max_length=max_len)["input_ids"])

            text_token['labels'][:, 1:prp_len + 1] = -100  # 占位，计算loss时忽略-100
            # print('text_token', text_token)
            return text_token

        train_dataset = train_dataset.map(preprocess_function, batched=True)
        test_dataset = test_dataset.map(preprocess_function, batched=True)
        return train_dataset, test_dataset

    # 构建trainer
    def create_trainer(self, model, train_dataset, test_dataset, checkpoint_dir, batch_size):
        args = TrainingArguments(
            checkpoint_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=20,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        def compute_metrics(pred):
            # labels = pred.label_ids
            # preds = pred.predictions.argmax(-1)
            labels = pred.label_ids[:, prp_len + 1]
            preds = pred.predictions[:, prp_len + 1].argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
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
    data_file = './data.txt'
    checkpoint_dir = "/checkpoint/"
    batch_size = 128
    max_len = 150
    n_label = 3
    tags_data = lct.data_show(data_file)
    text, label = lct.data_process(data_file)
    tokenizer, model = lct.create_model_tokenizer("raw_bert/bert-base-uncased")
    train_dataset, test_dataset = lct.create_dataset(text, label, tokenizer, max_len)
    trainer = lct.create_trainer(model, train_dataset, test_dataset, checkpoint_dir, batch_size)
    trainer.train()
    pred = trainer.predict(test_dataset)
    pred_label = np.argmax(pred[0][:, prp_len + 1], axis=1).tolist()
    true_label = pred[1][:, prp_len + 1].tolist()
    print(metrics.classification_report(true_label, pred_label))
    print(metrics.confusion_matrix(true_label, pred_label))


if __name__ == '__main__':
    main()
