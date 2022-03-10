import json

import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig, BertTokenizer
import numpy as np

config_path = "./checkpoint_fr_en_v1_2/checkpoint-532/config.json"
config = BertConfig.from_pretrained(config_path)  # 导入模型超参数
vocab_path = r"./raw_bert/bert-base-uncased/vocab.txt"
tokenizer = BertTokenizer.from_pretrained("./raw_bert/bert-base-uncased")
vocab_tab = {}
cnt = 0
with open(vocab_path, "r", encoding='utf-8') as f:
    for line in f.readlines():
        vocab_tab[cnt] = line[:-1]
        cnt = cnt + 1
vocab_tab[30522] = "cosmonaut"
vocab_tab[30523] = "sports team"
vocab_tab[30524] = "video game"
vocab_tab[30525] = "body of water"
classes = ['sportsman', 'singer', 'actor', 'politician', 'royal', 'cosmonaut', 'country', 'city', 'brand', 'airplane',
           'car', 'train', 'club', 'sports team', 'company', 'army', 'mall', 'school', 'hospital', 'airport', 'stadium',
           'government', 'body of water', 'mountain', 'park', 'island', 'movie', 'music', 'broadcast', 'video game',
           'war', 'disaster', 'competition', 'festival', 'language', 'award', 'disease']
label_ids = np.array([tokenizer.encode(l)[1] for l in classes])




class Bert_Model(nn.Module):
    def __init__(self, bert_path, config_file):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path, config=config_file)  # 加载预训练模型权重

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]

        return logit


model = Bert_Model(bert_path=r"./checkpoint_fr_en_v1_2/checkpoint-532/pytorch_model.bin", config_file=config)
maskpos = 3

model.eval()


def pre(text):
    encode_dict = tokenizer.encode_plus(text, max_length=150, padding='max_length', truncation=True)
    id = encode_dict["input_ids"]
    inputid = id[:]
    inputid[maskpos] = tokenizer.mask_token_id
    attid = encode_dict["attention_mask"]
    segmentid = encode_dict["token_type_ids"]
    inputid = torch.from_numpy(np.array([inputid])).long()
    attid = torch.from_numpy(np.array([attid])).long()
    segmentid = torch.from_numpy(np.array([segmentid])).long()
    out_test = model(inputid, attid, segmentid)
    tout_train_mask = out_test[:, 3, :]
    y_pred = tout_train_mask[:, label_ids]
    pos = y_pred.argmax(axis=1)[0]
    res = classes[pos]
    return res

cnt = 0
with open('./data/fr_en_v1/fr_comment_description.json', 'r', encoding='utf-8') as fr:
    tab = json.load(fr)
with open('./data/fr_en_v1/en_comment_description.json', 'r', encoding='utf-8') as fr:
    des_tab = json.load(fr)
des_tab.update(tab)
failed = []
res = {}

for obj in des_tab:
    try:
        text = 'entity is [MASK],' + des_tab[obj]['description'].replace(des_tab[obj]['ent_name'], "entity")
        ty = pre(text)
        print(obj, ty)
        res[obj] = ty
    except Exception as e:
        print(obj, "失败!")
        print(e)
        failed.append(obj)

with open("./data/fr_en_v1/hard_prompt_typing_result_2.json", "w", encoding="utf-8") as fw:
    json.dump(res, fw, sort_keys=False, ensure_ascii=False, indent=4)
print("failed...")
for obj in failed:
    print(obj)