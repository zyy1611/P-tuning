import heapq
import json
from openprompt.prompt_type import pre

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
top_k = 5   # 取top_k个作为样本


def read_probability_list():
    with open("../fr_en_v1/probability_files/probability_list.json", "r", encoding="utf-8") as fr:
        p_list = {}
        p_list_temp = json.load(fr)
        print(type(p_list_temp))
        for k, v in p_list_temp.items():
            p_list[k] = v[0]
        return p_list


def total_type_res(k):
    type_tab = {}
    with open("../fr_en_v1/typing_res/retype_typing_result_onto.json", 'r', encoding='utf8') as fr:
        json_data = json.load(fr)
        if k is True:
            for k, v in json_data.items():
                if type_tab.get(v) is None:
                    type_tab[v] = [k]
                else:
                    type_tab[v].append(k)
        else:
            type_tab = json_data
    print(len(type_tab))
    return type_tab


def get_probaility(commnet=False):
    if commnet is True:
        comment_tab = {}
        with open('../fr_en_v1/comment_descriptions/en_comment_description.json', 'r', encoding='utf-8') as fr:
            tab = json.load(fr)
            for obj in tab:
                comment_tab[obj] = tab[obj]['description'].replace(tab[obj]['ent_name'], "entity")

        with open('../fr_en_v1/comment_descriptions/fr_comment_description.json', 'r', encoding='utf-8') as fr:
            tab = json.load(fr)
            for obj in tab:
                comment_tab[obj] = tab[obj]['description'].replace(tab[obj]['ent_name'], "entity")

            return comment_tab
    with open('../fr_en_v1/probability_files/probability_max.json', 'r', encoding='utf-8') as fr:
        pro = json.load(fr)
        probability = sorted(pro.items(), key=lambda x: x[1], reverse=True)
    return probability


def get_onto_train():
    with open('../fr_en_v1/samples/data_onto.txt', 'r', encoding='utf-8') as fr:
        tab = json.load(fr)
    return tab


if __name__ == "__main__":
    type_res = total_type_res(False)  # 直接模板得到结果的分类结果
    type_tab = total_type_res(True)  # 直接模板得到结果的分类情况
    print(type_tab)

    probability = get_probaility()  # 每个实体能取到该type的概率
    comment_tab = get_probaility(commnet=True)
    onto_data = get_onto_train()  # 按照ontoEA训练集生成的训练集，已经按照取type概率进行纠正
    print("原始训练集大小:", len(onto_data))
    onto_data_set = set()
    onto_data_type_tab = {}  # 按照ontoEA训练集生成的训练集中每个type有多少个
    for obj in onto_data:
        cls = obj.strip().split('    ')[0]
        onto_data_set.add(cls)
        if onto_data_type_tab.get(cls) is None:
            onto_data_type_tab[cls] = 1
        else:
            onto_data_type_tab[cls] = onto_data_type_tab[cls] + 1
    add_ids = []
    for obj, v in probability:
        if onto_data_type_tab.get(type_res[obj]) is None or onto_data_type_tab.get(
                type_res[obj]) < 200:  # 如果是取top2，那么还要加另外的数据
            add_ids.append(obj)
            # onto_data.append(type_res[obj] + '    ' + comment_tab[obj])
            if onto_data_type_tab.get(type_res[obj]) is None:
                onto_data_type_tab[type_res[obj]] = 1
            else:
                onto_data_type_tab[type_res[obj]] = onto_data_type_tab[type_res[obj]] + 1
    p_list = read_probability_list()
    for id, p_some in p_list.items():
        if id not in add_ids:
            continue
        index_list = heapq.nlargest(top_k, range(len(p_some)), p_some.__getitem__)
        for i in range(top_k):
            onto_data.append(classes[index_list[i]] + '    ' + comment_tab[obj])
    print(onto_data_type_tab)
    print("扩充后训练集大小为:", len(onto_data))
    with open('../fr_en_v1/samples/data_onto_sampling.txt', 'w', encoding='utf-8') as fw:
        json.dump(onto_data, fw, sort_keys=False)
