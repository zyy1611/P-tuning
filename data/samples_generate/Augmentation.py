import copy
import json
import math
import random

random.seed(0)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3


def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
    words = sentence.split(' ')
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS) - 1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line


def main():
    dataset = "../fr_en_v1/samples/data_onto_sampling.txt"
    type_set = {}
    with open(dataset, 'r') as f:
        train_orig = json.load(f)
        print("原始训练数据集大小为:", len(train_orig))
        for line in train_orig:
            line1 = line.split('    ')
            label = line1[0]
            sentence = line1[1]
            if type_set.get(label) is None:
                type_set[label] = [sentence]
            else:
                type_set[label].append(sentence)

    res = copy.deepcopy(type_set)
    for obj in type_set:
        if len(type_set[obj]) in range(50, 201):
            cnt = 0
        else:
            cnt = 2
        print("{} 总计{}个句子，每个句子需要补充{}个：".format(obj, len(type_set[obj]), cnt))
        for sentence in type_set[obj]:
            # print("句子：{}".format(sentence))
            for i in range(cnt):
                sentence_aug = insert_punctuation_marks(sentence)
                # print("扩充第{}个：{}".format(i + 1, sentence_aug))
                line_aug = obj + '\t' + sentence_aug
                res[obj].append(line_aug)

    train_data = []
    for obj in res:
        for strr in res[obj]:
            train_data.append(obj + '    ' + strr)
    print("数据增强后大小为:", len(train_data))
    with open('../fr_en_v1/samples/data_onto_augmentation.txt', 'w', encoding='utf-8') as fw:
        json.dump(train_data, fw, sort_keys=False)


if __name__ == "__main__":
    main()
