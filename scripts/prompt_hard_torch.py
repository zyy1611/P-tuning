import json
import warnings

from torch.autograd import Variable

warnings.filterwarnings('ignore')
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import csv
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup, BertForMaskedLM
from datetime import datetime
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
classes = ['soccer player', 'singer', 'library', 'political party', 'department', 'economist', 'prime minister',
           'election', 'decoration', 'historic building', 'soccer manager', 'software', 'chancellor', 'tennis player',
           'football match', 'scientist', 'motorsport racer', 'village', 'athlete', 'painter', 'royalty', 'politician',
           'government agency', 'university', 'country', 'station', 'tennis tournament', 'gymnast', 'record label',
           'chess player', 'painting', 'play', 'soccer league', 'old territory', 'information appliance',
           'music festival', 'track list', 'prefecture', 'ice hockey player', 'overseas department',
           'formula one racer', 'building', 'architectural structure', 'governmental administrative region',
           'fictional character', 'airport', 'deputy', 'volcano', 'comics character', 'comics creator', 'school',
           'actor', 'sports event', 'baseball team', 'company', 'broadcast network', 'sea', 'ice hockey league', 'book',
           'wrestler', 'weapon', 'railway line', 'journalist', 'public transit system', 'classical music artist',
           'railway station', 'instrumentalist', 'mountain', 'lake', 'basketball team', 'plant', 'soccer tournament',
           'anime', 'screen writer', 'administrative region', 'ethnic group', 'state', 'disease', 'political function',
           'church', 'power station', 'chemical compound', 'ski area', 'settlement', 'military unit',
           'american football player', 'sport', 'vice president', 'song', 'military conflict', 'road', 'bank', 'award',
           'noble', 'monument', 'comedian', 'volleyball player', 'member of parliament', 'agent', 'album', 'senator',
           'golf tournament', 'colour', 'sports league', 'american football team', 'language', 'programming language',
           'governor', 'newspaper', 'racing driver', 'person', 'river', 'organisation', 'president', 'cricketer',
           'publisher', 'castle', 'museum', 'island', 'music genre', 'artwork', 'saint', 'insect', 'military person',
           'place', 'swimmer', 'olympic event', 'amateur boxer', 'musical artist', 'cleric', 'hollywood cartoon',
           'band', 'mayor', 'airline', 'office holder', 'hockey team', 'formula one team', 'basketball player', 'skier',
           'judge', 'city', 'academic journal', 'artist', 'christian bishop', 'film', 'baseball player',
           'mountain range', 'rugby club', 'radio station', 'television show', 'religious building', 'adult actor',
           'race track', 'grape', 'architect', 'soccer club', 'single', 'ski resort', 'cyclist', 'written work', 'town',
           'model', 'grand prix', 'entomologist', 'rugby player', 'dam', 'television station', 'musical', 'horse race',
           'archeologist', 'philosopher', 'boxer', 'skyscraper', 'musical work', 'protected area', 'manga', 'writer',
           'stadium', 'back scene', 'cardinal', 'video game']  # onto+video game 181个


# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)  # 这里看情况选择，如果之前softmax了，后续就不用了
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()

        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def read_data():
    text = []
    label = []
    with open("../data/fr_en_v1/samples/data_onto_augmentation.txt") as fr:
        json_data = json.load(fr)
        for line in json_data:
            line = line.strip().split('    ')
            text.append(line[1])
            label.append(line[0])
    return text, label


# 将你的数据和标签存入列表中
StrongData, StrongLabel = read_data()
print("数据集大小为:{}".format(len(StrongData)))
# 划分训练、测试、验证集
x_train, x_test, y_train, y_test = train_test_split(StrongData, StrongLabel, test_size=0.1, random_state=42)
# xp_train, xp_evil, yp_train, yp_evil = train_test_split(x_fune_train,y_fune_train, test_size=0.2, random_state=42)
print("训练数据共:{},测试数据共:{}".format(len(x_train), len(x_test)))

# 导入Bert分词器
# vocab_path = r"./raw_bert/bert-base-uncased/vocab.txt"
# tokenizer = BertTokenizer.from_pretrained(vocab_path)
tokenizer = BertTokenizer.from_pretrained("../raw_bert/bert-base-uncased")
len_tok = len(tokenizer)

# 模板
prefix = 'entity was Mask. '
type_id_tab = {}
for obj in classes:
    type_id_tab[obj] = tokenizer.convert_tokens_to_ids(obj)
# 你的空处填的词
# pos_id = tokenizer.convert_tokens_to_ids('good')  # 2204
# neg_id = tokenizer.convert_tokens_to_ids('bad')  # 2919

# 构建训练集
Inputid = []
Labelid = []
sid = []
atid = []

for i in range(len(x_train)):
    text_ = prefix + x_train[i]
    encode_dict = tokenizer.encode_plus(text_, max_length=100, padding='max_length', truncation=True)

    id = encode_dict["input_ids"]
    segmentid = encode_dict["token_type_ids"]
    attid = encode_dict["attention_mask"]
    labelid, inputid = id[:], id[:]
    maskpos = 3
    type_id = type_id_tab[y_train[i]]
    labelid[maskpos] = type_id
    labelid[: maskpos] = [-1] * len(labelid[: maskpos])
    labelid[maskpos + 1:] = [-1] * len(labelid[maskpos + 1:])
    inputid[maskpos] = tokenizer.mask_token_id
    # if y_train[i] == 0:
    #     labelid[maskpos] = neg_id
    #     labelid[: maskpos] = [-1] * len(labelid[: maskpos])
    #     labelid[maskpos + 1:] = [-1] * len(labelid[maskpos + 1:])
    #     inputid[maskpos] = tokenizer.mask_token_id
    # else:
    #     labelid[maskpos] = pos_id
    #     labelid[: maskpos] = [-1] * len(labelid[: maskpos])
    #     labelid[maskpos + 1:] = [-1] * len(labelid[maskpos + 1:])
    #     inputid[maskpos] = tokenizer.mask_token_id

    Labelid.append(labelid)
    Inputid.append(inputid)
    sid.append(segmentid)
    atid.append(attid)

Inputid = np.array(Inputid)
Labelid = np.array(Labelid)
sid = np.array(sid)
atid = np.array(atid)

print(Inputid.shape)
print(Labelid.shape)

print("正在划分数据集")
idxes = np.arange(Inputid.shape[0])  # idxes的第一维度，也就是数据大小
np.random.seed(2022)  # 固定种子
np.random.shuffle(idxes)
a = 4509
# 划分训练集、验证集
# input_ids_train, input_ids_valid = Inputid[idxes[:a]], Inputid[idxes[a:5632]]
# input_masks_train, input_masks_valid = atid[idxes[:a]], atid[idxes[a:5632]]
# input_types_train, input_types_valid = sid[idxes[:a]], sid[idxes[a:5632]]
# label_train, y_valid = Labelid[idxes[:a]], Labelid[idxes[a:5632]]
# print(input_ids_train.shape, label_train.shape, input_ids_valid.shape, y_valid.shape)
# print(label_train[:, 3])

# 测试集构建
tInputid = []
tLabelid = []
tsid = []
tatid = []
for i in range(len(x_test)):
    text_ = prefix + x_test[i]
    encode_dict = tokenizer.encode_plus(text_, max_length=100, padding='max_length', truncation=True)
    id = encode_dict["input_ids"]
    segmentid = encode_dict["token_type_ids"]
    attid = encode_dict["attention_mask"]
    labelid, inputid = id[:], id[:]
    maskpos = 3
    type_id = type_id_tab[y_test[i]]
    labelid[maskpos] = type_id
    labelid[: maskpos] = [-1] * len(labelid[: maskpos])
    labelid[maskpos + 1:] = [-1] * len(labelid[maskpos + 1:])
    inputid[maskpos] = tokenizer.mask_token_id
    # if y_test[i] == 0:
    #     labelid[maskpos] = neg_id
    #     labelid[: maskpos] = [-1] * len(labelid[: maskpos])
    #     labelid[maskpos + 1:] = [-1] * len(labelid[maskpos + 1:])
    #     inputid[maskpos] = tokenizer.mask_token_id
    # else:
    #     labelid[maskpos] = pos_id
    #     labelid[: maskpos] = [-1] * len(labelid[: maskpos])
    #     labelid[maskpos + 1:] = [-1] * len(labelid[maskpos + 1:])
    #     inputid[maskpos] = tokenizer.mask_token_id

    tLabelid.append(labelid)
    tInputid.append(inputid)
    tsid.append(segmentid)
    tatid.append(attid)

tInputid = np.array(tInputid)
tLabelid = np.array(tLabelid)
tsid = np.array(tsid)
tatid = np.array(tatid)
print("测试集大小", tInputid.shape, tLabelid.shape)


# 构建数据集


class MyDataSet(Data.Dataset):
    def __init__(self, sen, mask, typ, label):
        super(MyDataSet, self).__init__()
        self.sen = sen
        self.mask = mask
        self.typ = typ
        self.label = label

    def __len__(self):
        return self.sen.shape[0]

    def __getitem__(self, idx):
        return self.sen[idx], self.mask[idx], self.typ[idx], self.label[idx]


input_ids_train = torch.from_numpy(Inputid).long()
# input_ids_train = torch.from_numpy(input_ids_train).long()
# input_ids_valid = torch.from_numpy(input_ids_valid).long()
input_ids_test = torch.from_numpy(tInputid).long()

input_masks_train = torch.from_numpy(atid).long()
# input_masks_train = torch.from_numpy(input_masks_train).long()
# input_masks_valid = torch.from_numpy(input_masks_valid).long()
input_masks_test = torch.from_numpy(tatid).long()

input_types_train = torch.from_numpy(sid).long()
# input_types_train = torch.from_numpy(input_types_train).long()
# input_types_valid = torch.from_numpy(input_types_valid).long()
input_types_test = torch.from_numpy(tsid).long()

label_train = torch.from_numpy(Labelid).long()
# y_valid = torch.from_numpy(y_valid).long()
label_test = torch.from_numpy(tLabelid).long()

train_dataset = Data.DataLoader(MyDataSet(input_ids_train, input_masks_train, input_types_train, label_train), 32, True)
# valid_dataset = Data.DataLoader(MyDataSet(input_ids_valid, input_masks_valid, input_types_valid, y_valid), 32, True)
test_dataset = Data.DataLoader(MyDataSet(input_ids_test, input_masks_test, input_types_test, label_test), 128, True)


# 构建模型
class Bert_Model(nn.Module):
    def __init__(self, bert_path):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path)  # 加载预训练模型权重
        self.bert.resize_token_embeddings(len_tok)  # !!!

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]

        return logit


# config_path = r"./raw_bert/bert-base-uncased/config.json"
# config = BertConfig.from_pretrained(config_path)  # 导入模型超参数
# print(config)


# bert_path=r"/mnt/JuChiYun-WangFei/bert-base-chinese/pytorch_model.bin"
print("正在加载模型")
model = Bert_Model(bert_path=r"../raw_bert/bert-base-uncased").to(
    DEVICE)
print("模型加载完毕")

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # 使用Adam优化器
# loss_func = nn.CrossEntropyLoss(ignore_index=-1)  # 换成facal loss！！！
loss_func = FocalLoss(num_class=len_tok)
EPOCH = 200
schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset),
                                           num_training_steps=EPOCH * len(train_dataset))
print("正在训练中。。。")
import time

for epoch in range(EPOCH):

    starttime_train = datetime.now()
    start = time.time()
    correct = 0
    train_loss_sum = 0.0
    model.train()
    print("***** Running training epoch {} *****".format(epoch + 1))

    for idx, (ids, att, tpe, y) in enumerate(train_dataset):
        ids, att, tpe, y = ids.to(DEVICE), att.to(DEVICE), tpe.to(DEVICE), y.to(DEVICE)
        out_train = model(ids, att, tpe)
        # print(out_train.view(-1, len_tok).shape, y.view(-1).shape)
        loss = loss_func(out_train[:, 3, :].view(-1, len_tok), y[:, 3].view(-1))
        # loss = loss_func(out_train,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        train_loss_sum += loss.item()

        if (idx + 1) % 40 == 0:
            print("Epoch {:04d} | Step {:06d}/{:06d} | Loss {:.4f} | Time {:.0f}".format(
                epoch + 1, idx + 1, len(train_dataset), train_loss_sum / (idx + 1), time.time() - start))

        truelabel = y[:, 3]
        out_train_mask = out_train[:, 3, :]

        predicted = torch.max(out_train_mask.data, 1)[1]
        correct += (predicted == truelabel).sum()
        correct = np.float(correct)
    acc = float(correct / len(label_train))

    eval_loss_sum = 0.0
    model.eval()
    correct_test = 0
    with torch.no_grad():
        for ids, att, tpe, y in test_dataset:
            ids, att, tpe, y = ids.to(DEVICE), att.to(DEVICE), tpe.to(DEVICE), y.to(DEVICE)
            out_test = model(ids, att, tpe)
            loss_eval = loss_func(out_test[:, 3, :].view(-1, len_tok), y[:, 3].view(-1))
            # loss_eval = loss_func(out_train, y)
            eval_loss_sum += loss_eval.item()
            ttruelabel = y[:, 3]
            tout_train_mask = out_test[:, 3, :]
            predicted_test = torch.max(tout_train_mask.data, 1)[1]
            correct_test += (predicted_test == ttruelabel).sum()
            correct_test = np.float(correct_test)
    acc_test = float(correct_test / len(label_test))

    if epoch % 1 == 0:
        out = ("epoch {}, train_loss {},  train_acc {} , eval_loss {} ,acc_test {}"
               .format(epoch + 1, train_loss_sum / (len(train_dataset)), acc, eval_loss_sum / (len(test_dataset)),
                       acc_test))

        print(out)
torch.save(model, "../models/torch_models/torch_hard_facal_loss_augmentation/model_top3.pth")
