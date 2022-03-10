import json


def type_count():
    cnt_1 = {}
    cnt_2 = {}
    with open("./fr_en_v1/hard_prompt_typing_result.json", 'r', encoding='utf8') as fr:
        json_data = json.load(fr)
        for obj in json_data:
            if cnt_1.get(json_data[obj]) is None:
                cnt_1[json_data[obj]] = 1
            cnt_1[json_data[obj]] = cnt_1[json_data[obj]] + 1
        cnt_1 = sorted(cnt_1.items(), key=lambda x: x[1], reverse=True)
    print(cnt_1)
    print(len(cnt_1))
    # with open("./fr_en_v1/fr_typing_result.json", 'r', encoding='utf8') as fr:
    #     json_data = json.load(fr)
    #     for obj in json_data:
    #         if cnt_1.get(json_data[obj]) is None:
    #             cnt_1[json_data[obj]] = 1
    #         cnt_1[json_data[obj]] = cnt_1[json_data[obj]] + 1
    #     cnt_1 = sorted(cnt_1.items(), key=lambda x: x[1], reverse=True)
    # print(cnt_1)
    # with open("./fr_en_v1/en_typing_result.json", 'r', encoding='utf8') as fr:
    #     json_data = json.load(fr)
    #     for obj in json_data:
    #         if cnt_2.get(json_data[obj]) is None:
    #             cnt_2[json_data[obj]] = 1
    #         cnt_2[json_data[obj]] = cnt_2[json_data[obj]] + 1
    #     cnt_2 = sorted(cnt_2.items(), key=lambda x: x[1], reverse=True)
    # print(cnt_2)


def hit_compute():
    with open("./fr_en_v1/ref_ent_ids", 'r', encoding='utf8') as fr:
        ret = []
        for line in fr:
            ret.append(line[:-1].split("\t"))
    #     with open("./fr_en_v1/fr_typing_result.json", 'r', encoding='utf8') as fr:
    #         tab = json.load(fr)
    #     with open("./fr_en_v1/en_typing_result.json", 'r', encoding='utf8') as fr:
    #         json_data = json.load(fr)
    #         for obj in json_data:
    #             tab[obj] = json_data[obj]
        with open("./fr_en_v1/hard_prompt_typing_result.json", 'r', encoding='utf8') as fr:
            tab = json.load(fr)
        print("total typing..",len(tab))
        cls_1 = {}
        cls_2 = {}
        for obj in ret:
            left = obj[0]
            right = obj[1]
            if cls_1.get(tab[left]) is None:
                cls_1[tab[left]] = 1
            cls_1[tab[left]] = cls_1[tab[left]] + 1
            if cls_2.get(tab[right]) is None:
                cls_2[tab[right]] = 1
            cls_2[tab[right]] = cls_2[tab[right]] + 1
        cls_1 = sorted(cls_1.items(), key=lambda x: x[1], reverse=True)
        cls_2 = sorted(cls_2.items(), key=lambda x: x[1], reverse=True)
        print(cls_1)
        print(cls_2)

        cnt = 0
        for obj in ret:
            if tab[obj[0]] == tab[obj[1]]:
                cnt = cnt + 1
        print(cnt)


if __name__ == "__main__":
    type_count()
    hit_compute()
