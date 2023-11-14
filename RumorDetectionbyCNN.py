# -*- coding: utf-8 -*-

'''
@article{song2018ced,
# title={CED: Credible Early Detection of Social Media Rumors},
#  author={Song, Changhe and Tu, Cunchao and Yang, Cheng and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:1811.04175},
  year={2018}
'''

import zipfile
import os
import io
import random
import json
import pandas as pd

# import matplotlib.pyplot as plt
# import numpy as np
# import paddle
# import paddle.fluid as fluid
# import paddle.nn as nn
# from paddle.nn import Conv2D, Linear, Embedding
# from paddle.fluid.dygraph.base import to_variable


# 解压原始数据集，将Rumor_Dataset.zip解压至data目录下
# Unzip the original dataset and extract it to the target directory
src_path = "C:/Users/杨思博/Downloads/Rumor_Dataset.zip"  # 这里填写自己项目所在的数据集路径
target_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master"
if (not os.path.isdir(target_path)):
    z = zipfile.ZipFile(src_path, 'r')
    z.extractall(path=target_path)
    z.close()

# 分别为谣言数据、非谣言数据、全部数据的文件路径
# Define paths for rumor 、non-rumor and original_data datasets
rumor_class_dirs = os.listdir(
    os.path.join(target_path, "C:/users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost"))
# 这里填写自己项目所在的数据集路径
non_rumor_class_dirs = os.listdir(
    os.path.join(target_path, "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost"))
original_microblog = os.path.join(target_path,
                                  "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog")

# 谣言标签为0，非谣言标签为1
rumor_label = "0"
non_rumor_label = "1"

# 分别统计谣言数据与非谣言数据的总数
rumor_num = 0
non_rumor_num = 0
all_rumor_list = []
all_non_rumor_list = []

# 解析谣言数据 analyse rumor
for rumor_class_dir in rumor_class_dirs:
    if not rumor_class_dir.endswith('.DS_Store'):
        # 遍历谣言数据，并解析
        file_path = os.path.join(original_microblog, rumor_class_dir)
        with open(file_path, 'r', encoding='utf-8') as f:
            rumor_content = f.read()
        rumor_dict = json.loads(rumor_content)
        all_rumor_list.append(rumor_label + "\t" + rumor_dict["text"] + "\n")
        rumor_num += 1
# 解析非谣言数据 analyse non_rumor
for non_rumor_class_dir in non_rumor_class_dirs:
    if not non_rumor_class_dir.endswith('.DS_Store'):
        file_path = os.path.join(original_microblog, non_rumor_class_dir)
        with open(file_path, 'r', encoding='utf-8') as f2:
            non_rumor_content = f2.read()
        non_rumor_dict = json.loads(non_rumor_content)
        all_non_rumor_list.append(non_rumor_label + "\t" + non_rumor_dict["text"] + "\n")
        non_rumor_num += 1

print("The total amount of rumor data ：" + str(rumor_num))
print("The total amount of non_rumor data：" + str(non_rumor_num))

# 全部数据进行乱序后写入all_data.txt
# Shuffle and write all_data.txt
data_list_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/"

all_data_path = data_list_path + "all_data.txt"
all_data_list = all_rumor_list + all_non_rumor_list

random.shuffle(all_data_list)

# 在生成all_data.txt之前，首先将其清空
with open(all_data_path, 'w', encoding='utf-8') as f:
    f.seek(0)
    f.truncate()

with open(all_data_path, 'a', encoding='utf-8') as f:
    for data in all_data_list:
        f.write(data)
print('all_data.txt generated!')


# 生成数据字典  dict.txt
# generate the dict.txt
# 生成数据字典
import string


def create_dict(data_path, dict_path):
    dict_count = {}

    # 获取标点符号集合
    punctuation_set = set(string.punctuation)

    # 读取全部数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 统计每个字符的出现次数（排除标点符号）
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            if s not in punctuation_set:  # 检查字符是否不是标点符号
                if s in dict_count:
                    dict_count[s] += 1
                else:
                    dict_count[s] = 1

    # 把这些字符及其出现次数保存到本地的 dict.txt 文件中
    with open(dict_path, 'w', encoding='utf-8') as f:
        for key, value in dict_count.items():
            f.write(f"{key}: {value}\n")

    print("Data dictionary generation completed！", '\t', 'The length of the dictionary：', len(dict_count))


data_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/all_data.txt"
dict_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/dict.txt"
create_dict(data_path, dict_path)


# Create serialized representation of data and split into training and evaluation sets
# The purpose of this code is to process raw data into data lists for training and  evaluation of text classification models.
# There are TWO file generated：eval_list.txt is for evalution，train_list.txt is foe training。
# 创建序列化表示的数据,并按照一定比例划分训练数据与验证数据
def create_data_list(data_list_path):
    # 读取包含字典数据的文件，并将内容解析为Python字典
    dict_txt = {}
    with open(os.path.join(data_list_path, 'dict.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()
        for line in lines:
            key, value = line.strip().split(': ')
            dict_txt[key] = int(value)

    with open(os.path.join(data_list_path, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    i = 0
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'a', encoding='utf-8') as f_eval, \
            open(os.path.join(data_list_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
        for line in lines:
            title = line.split('\t')[-1].replace('\n', '')
            lab = line.split('\t')[0]
            t_ids = ""

            if i % 8 == 0:
                for s in title:
                    # 检查字典中是否存在键 s，如果存在则获取其对应的值，否则返回一个默认值（例如空字符串）
                    temp = str(dict_txt.get(s, ''))
                    t_ids = t_ids + temp + ','
                t_ids = t_ids[:-1] + '\t' + lab + '\n'
                f_eval.write(t_ids)

            else:
                for s in title:
                    # 检查字典中是否存在键 s，如果存在则获取其对应的值，否则返回一个默认值（例如 -1）
                    temp = str(dict_txt.get(s, -1))
                    # 如果字典中不存在键 s，temp 将会是 -1 或你指定的默认值
                    # 进行后续处理，例如输出提示信息或根据需要执行其他操作
                    if temp == -1:
                        print(f"The character '{s}' was not found in the dictionary.")
                    # 其他情况下，temp 将是字典中键 s 对应的值
                    else:
                        t_ids = t_ids + temp + ','
            i += 1

    print("数据列表生成完成！")


data_list_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/"
create_data_list(data_list_path)


# This code defines a data reader function, data_reader, to read data from a specified file and return a data generator function.
# The generator function reads data line by line from the file, parses text content and labels, and returns them.
# If the shuffle parameter is specified and the phrase is "train", the data is shuffled for training.
# This data reader can be used to load training and evaluation data for text classification tasks.
def data_reader(file_path, phrase, shuffle=False):
    # 初始化一个空列表用于存储数据
    all_data = []

    # 打开文件并逐行读取数据
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            # 分割每行数据，按照制表符分隔
            cols = line.strip().split("\t")
            # 如果数据列数不等于2，跳过该行数据
            if len(cols) != 2:
                continue
            # 解析标签，将第二列数据转换为整数
            label = int(cols[1])

            # 分割文本内容，将第一列数据按逗号分隔并存储为列表
            wids = cols[0].split(",")
            # Combine the text content and label into a tuple and add it to the list of data.
            all_data.append((wids, label))

    # 如果需要对数据进行洗牌（shuffle）
    if shuffle:
        if phrase == "train":
            random.shuffle(all_data)

    # 定义一个数据生成器函数
    def reader():
        for doc, label in all_data:
            # 生成器函数，每次迭代产生一组文本内容和标签
            yield doc, label

    # 返回数据生成器函数
    return reader


file_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/train_list.txt"
phrase = "train"  # or "eval", indicating whether the dataset is for training or evaluation  或 "eval"，表示数据集是训练集还是评估集
shuffle = True  # hether to shuffle the training data  是否对训练数据进行洗牌

# 调用数据读取器函数，返回一个数据生成器
# Call the data reader function, returning a data generator
data_generator = data_reader(file_path, phrase, shuffle)
