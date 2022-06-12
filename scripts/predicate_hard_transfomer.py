import json

import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig, BertTokenizer
import numpy as np

model_path = "./checkpoint_fr_en_v1_onto/checkpoint-1053"
config_path = "{}/config.json".format(model_path)
config = BertConfig.from_pretrained(config_path)  # 导入模型超参数
vocab_path = r"./raw_bert/bert-base-uncased/vocab.txt"
tokenizer = BertTokenizer.from_pretrained("./raw_bert/bert-base-uncased")
vocab_tab = {}
cnt = 0
# classes = ['sportsman', 'singer', 'actor', 'politician', 'royal', 'cosmonaut', 'country', 'city', 'brand', 'airplane',
#            'car', 'train', 'club', 'sports team', 'company', 'army', 'mall', 'school', 'hospital', 'airport', 'stadium',
#            'government', 'body of water', 'mountain', 'park', 'island', 'movie', 'music', 'broadcast', 'video game',
#            'war', 'disaster', 'competition', 'festival', 'language', 'award', 'disease']  # manual
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
with open(vocab_path, "r", encoding='utf-8') as f:
    for line in f.readlines():
        vocab_tab[cnt] = line[:-1]
        cnt = cnt + 1
# new_tokens = ['cosmonaut', 'sports team', 'video game', 'body of water']  #maual
new_tokens = {"cosmonaut": 30522, "sports team": 30523, "video game": 30524, "body of water": 30525,
              "american football team": 30526, "ski resort": 30527, "screen writer": 30528, "radio station": 30529,
              "soccer player": 30530, "basketball player": 30531, "power station": 30532, "hockey team": 30533,
              "race track": 30534, "christian bishop": 30535, "soccer league": 30536, "hollywood cartoon": 30537,
              "sports league": 30538, "tennis tournament": 30539, "grand prix": 30540, "ski area": 30541,
              "written work": 30542, "soccer club": 30543, "military conflict": 30544, "formula one racer": 30545,
              "office holder": 30546, "political party": 30547, "adult actor": 30548, "soccer tournament": 30549,
              "architectural structure": 30550, "amateur boxer": 30551, "chess player": 30552,
              "political function": 30553, "record label": 30554, "fictional character": 30555, "music festival": 30556,
              "track list": 30557, "musical artist": 30558, "formula one team": 30559, "back scene": 30560,
              "television show": 30561, "music genre": 30562, "administrative region": 30563, "racing driver": 30564,
              "historic building": 30565, "american football player": 30566, "academic journal": 30567,
              "public transit system": 30568, "prime minister": 30569, "ethnic group": 30570, "vice president": 30571,
              "protected area": 30572, "comics character": 30573, "ice hockey player": 30574, "rugby player": 30575,
              "olympic event": 30576, "television station": 30577, "volleyball player": 30578, "military person": 30579,
              "railway line": 30580, "government agency": 30581, "chemical compound": 30582,
              "programming language": 30583, "motorsport racer": 30584, "football match": 30585,
              "religious building": 30586, "entomologist": 30587, "comics creator": 30588, "tennis player": 30589,
              "railway station": 30590, "rugby club": 30591, "archeologist": 30592, "basketball team": 30593,
              "governmental administrative region": 30594, "old territory": 30595, "member of parliament": 30596,
              "horse race": 30597, "golf tournament": 30598, "ice hockey league": 30599, "information appliance": 30600,
              "overseas department": 30601, "musical work": 30602, "military unit": 30603, "baseball team": 30604,
              "soccer manager": 30605, "classical music artist": 30606, "mountain range": 30607,
              "broadcast network": 30608, "baseball player": 30609, "sports event": 30610,
              "video game": 30611}
for k, v in new_tokens.items():
    vocab_tab[v] = k
print(len(vocab_tab))
label_ids = np.array([tokenizer.encode(l)[1] for l in classes])


class Bert_Model(nn.Module):
    def __init__(self, bert_path, config_file):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path, config=config_file)  # 加载预训练模型权重

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]

        return logit


model = Bert_Model(bert_path=r"{}/pytorch_model.bin".format(model_path), config_file=config)
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
    print(inputid, attid, segmentid)
    out_test = model(inputid, attid, segmentid)
    print(out_test.shape)
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

with open("./data/fr_en_v1/hard_prompt_typing_result_onto_from_ontoEA_augmentation.json", "w", encoding="utf-8") as fw:
    json.dump(res, fw, sort_keys=False, ensure_ascii=False, indent=4)
print("failed...")
for obj in failed:
    print(obj)
