# -*- coding: utf-8 -*-
# @Date    : 2020/11/4
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : ccf_2020_qa_match_pet.py
"""
Pattern-Exploiting Training(PET): 增加pattern，将任务转换为MLM任务。
线上f1: 0.761

tips:
  切换模型时，修改对应config_path/checkpoint_path/dict_path路径以及build_transformer_model 内的参数
"""

import os
import numpy as np
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import *
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# PET-文本分类的又一种妙解:https://xv44586.github.io/2020/10/25/pet/
# ccf问答匹配比赛（下）：如何只用“bert”夺冠:https://xv44586.github.io/2021/01/20/ccf-qa-2/
num_classes = 32
maxlen = 128
batch_size = 8

# BERT base

config_path = 'data/pretrained/nezha/NEZHA-Base/bert_config.json'
checkpoint_path = 'data/pretrained/nezha/NEZHA-Base/model.ckpt-900000'
dict_path = 'data/pretrained/nezha/NEZHA-Base/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)

# pattern
pattern = '下面两个句子的语义相似度较高:'
# tokenizer.encode的第一个位置是cls，所以mask的index要+1
tokens = ["CLS"]+list(pattern)
print(tokens[14])
mask_idx = [14]

id2label = {
    0: '低',
    1: '高'
}

label2id = {v: k for k, v in id2label.items()}
print('label2id:',label2id)#label2id: {'低': 0, '高': 1}
labels = list(id2label.values())
print('labels:',labels)#labels: ['低', '高']
# labels在token中的ids,encode的时候，第一个数是cls，所以取encode输出的tokens[1:-1]，代表跳过了cls的
label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])
print('label_ids:',label_ids)#label_ids: [[ 856] [7770]]

# 这里本文其实没有用到
def random_masking(token_ids):
    """对输入进行随机mask
    """
    # n个随机数
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        # [mask, 0.15 * 0.8, t(本身), 0.15 * 0.9, 随机, 0.15, 本身，target=0，其余target都为1]
        if r < 0.15 * 0.8:
            # 通过mask来预测target
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            # 通过本身来预测target
            source.append(t)
            target.append(t)
        elif r < 0.15:
            # 通过随机token来预测target
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            # 通过本身->label=0?
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    def __init__(self, prefix=False, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.prefix = prefix

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_target_ids = [], [], []
        # 拿到query和reply
        for is_end, (q, r, label) in self.get_sample(shuffle):
            # 没有label的时候定义为None
            label = int(label) if label is not None else None
            # 有label的时候，才添加前缀
            if label is not None or self.prefix:
                q = pattern + q
            # 拿到token_ids和segment_id
            token_ids, segment_ids = tokenizer.encode(q, r, maxlen=maxlen)
            # 本文没有用到这个
            if shuffle:
                # 这里做了随机mask，随机mask有点没看懂, 但是本文都没用到这个
                source_tokens, target_tokens = random_masking(token_ids)
            else:
                # 理论上target_tokens就等于source_tokens
                source_tokens, target_tokens = token_ids[:], token_ids[:]
            # mask label
            if label is not None:
                # 将label转化成token，因为是mlm任务，最终的label其实就是token
                label_ids = tokenizer.encode(id2label[label])[0][1:-1]
                # pattern = '直接回答问题:'
                # mask_idx = [1]
                # 这里label_ids也只有一个，所以是直接复制
                # mask_idx代表的其实是label在原文中的位置
                for m, lb in zip(mask_idx, label_ids):
                    # 这里相当于把原文的label更换成为mask_id
                    # source_tokens[1] = mask_id
                    # 然后target_tokens[1] = label_id(也就是label对应的token_id)
                    # 这里只更改了label对应的token，其余部分不变
                    source_tokens[m] = tokenizer._token_mask_id
                    target_tokens[m] = lb
            elif self.prefix:
                # 这里就一个mask_id，如果有多个多个都直接赋值成为token_id
                for i in mask_idx:
                    source_tokens[i] = tokenizer._token_mask_id
            # 最后拿到mlm任务的source_tokens,segment_ids,target_tokens
            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)

            if is_end or len(batch_token_ids) == self.batch_size:
                # 满足batch_size要求了，把他yield出去
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_target_ids = pad_sequences(batch_target_ids)
                # batch_target_ids是每个位置target的id
                yield [batch_token_ids, batch_segment_ids, batch_target_ids], None
                # 将原始的batch里面的内容置为空
                batch_token_ids, batch_segment_ids, batch_target_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        # K.not_equal, 拿到y_true不为0的部分，然后转化成为float
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        # 计算精度
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        # mask掉输入部分
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        # 拿到acc精度
        self.add_metric(accuracy, name='accuracy')
        # 拿到交叉熵
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        # mask
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# tokenizer
# tokenizer = Tokenizer(dict_path, do_lower_case=True)



def train(train_data, val_data, test_data, best_model_file, test_result_file):
    train_generator = data_generator(data=train_data + test_data, batch_size=batch_size)
    valid_generator = data_generator(data=val_data, batch_size=batch_size)
    test_generator = data_generator(data=test_data, batch_size=batch_size, prefix=True)
    target_in = Input(shape=(None,))
    model = build_transformer_model(config_path=config_path,
                                    checkpoint_path=checkpoint_path,
                                    with_mlm=True,  # with_nlm为True是不是返回的output就不一样了，应该返回的就是mlm的output
                                    # model='bert',  # 加载bert/Roberta/ernie
                                    model='nezha'
                                    )
    output = CrossEntropy(output_idx=1)([target_in, model.output])
    # 输入的时候，添加一个target_in， 输出还是和之前一样
    train_model = Model(model.inputs + [target_in], output)
    # 梯度衰减+梯度积累
    AdamW = extend_with_weight_decay(Adam)
    AdamWG = extend_with_gradient_accumulation(AdamW)
    opt = AdamWG(learning_rate=1e-5, exclude_from_weight_decay=['Norm', 'bias'], grad_accum_steps=4)
    train_model.compile(opt)
    train_model.summary()

    def evaluate(data):
        P, R, TP = 0., 0., 0.
        for d, _ in tqdm(data):
            x_true, y_true = d[:2], d[2]
            # 拿到预测结果，已经转化为label_ids里面的index了
            y_pred = predict(x_true)
            # 只取mask_idx对应的y -> 原始token -> 原始label中的index
            y_true = np.array([labels.index(tokenizer.decode(y)) for y in y_true[:, mask_idx]])
            # print(y_true, y_pred)
            # 计算f1
            R += y_pred.sum()
            P += y_true.sum()
            TP += ((y_pred + y_true) > 1).sum()
        print(P, R, TP)
        pre = TP / R
        rec = TP / P
        return 2 * (pre * rec) / (pre + rec)

    def predict(x):
        if len(x) == 3:
            x = x[:2]
        # 拿到mask_idx对应的output
        # todo:这里这个model为什么不是train_model啊?
        y_pred = model.predict(x)[:, mask_idx]
        # 这个维度信息不太清楚
        # batch, 0,label_ids对应的值, label_ids应该是可能有多个id，对应分类的多个类别
        y_pred = y_pred[:, 0, label_ids[:, 0]]
        # 最后是取得所有label_ids里面的最大值，得到mlm的预测结果的，这里面的mlm的预测的结果的个数与分类的label数一致
        y_pred = y_pred.argmax(axis=1)
        return y_pred

    class Evaluator(keras.callbacks.Callback):
        def __init__(self, valid_generator, best_pet_model_file="best_pet_model.weights"):
            self.best_acc = 0.
            self.valid_generator = valid_generator
            self.best_pet_model_file = best_pet_model_file

        def on_epoch_end(self, epoch, logs=None):
            acc = evaluate(self.valid_generator)
            if acc > self.best_acc:
                self.best_acc = acc
                self.model.save_weights(self.best_pet_model_file)
            print('acc :{}, best acc:{}'.format(acc, self.best_acc))

    def write_to_file(path, test_generator, test_data):
        preds = []
        # 分批预测结果
        for x, _ in tqdm(test_generator):
            pred = predict(x)
            preds.extend(pred)

        # 把原始的query，reply以及预测的p都写入到文件中
        ret = []
        for data, p in zip(test_data, preds):
            if data[2] is None:
                label = -1
            else:
                label = data[2]
            ret.append([data[0], data[1], str(label), str(p)])

        with open(path, 'w') as f:
            for r in ret:
                f.write('\t'.join(r) + '\n')

    evaluator = Evaluator(valid_generator, best_model_file)
    train_model.fit_generator(train_generator.generator(),
                              steps_per_epoch=len(train_generator),
                              epochs=10,
                              callbacks=[evaluator])

    train_model.load_weights(best_model_file)
    write_to_file(test_result_file, test_generator, test_data)

def load_pair_data(f, isshuffle=False):
    data = []
    df = pd.read_csv(f)
    if isshuffle:
        df = df.sample(frac=1.0, random_state=1234)
    columns = list(df.columns)
    if 'text_a' not in columns and 'query1' in columns:
        df.rename(columns={'query1':'text_a', 'query2':'text_b'}, inplace=True)
    for i in range(len(df)):
        can = df.iloc[i]
        text_a = can['text_a']
        text_b = can['text_b']
        if 'label' not in columns:
            label = None
        else:
            label = int(can['label'])
            if label == -1:
                label = None
        data.append([text_a, text_b, label])
    return data

def load_data():
    """
    :return: [text_a, text_b, label]
    天池疫情文本匹配数据集
    """
    data_dir = '../data/tianchi/'
    train_file = data_dir + 'train_20200228.csv'
    dev_file = data_dir + 'dev_20200228.csv'
    test_file = data_dir + 'test.example_20200228.csv'
    train_data = load_pair_data(train_file)
    val_data = load_pair_data(dev_file)
    test_data = load_pair_data(test_file)
    return train_data, val_data, test_data


def test_data_generator():
    data_dir = '../data/tianchi/'
    train_file = data_dir + 'train_20200228.csv'
    data = load_pair_data(train_file)
    train_generator = data_generator(data=data, batch_size=batch_size)
    for d in train_generator:
        print(d)
        break

def run():
    train_data, val_data, test_data = load_data()
    best_model_file = 'best_pet_model.weights'
    test_result_file = 'pet_submission.tsv'
    train(train_data, val_data, test_data, best_model_file, test_result_file)


if __name__ == '__main__':
    test_data_generator()
    run()



