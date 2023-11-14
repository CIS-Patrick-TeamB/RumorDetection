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
src_path = "C:/Users/杨思博/Downloads/Rumor_Dataset.zip"  # 这里填写自己数据集的zip文件所在的路径
target_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master" # zip解压后，Chinese_Rumor_Dataset-master文件夹所在地
if (not os.path.isdir(target_path)):
    z = zipfile.ZipFile(src_path, 'r')
    z.extractall(path=target_path)
    z.close()

# 分别为谣言数据、非谣言数据、全部数据的文件路径
# Define paths for rumor 、non-rumor and original_data datasets
rumor_class_dirs = os.listdir(
    os.path.join(target_path, "C:/users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost"))# rumor-repost的路径

non_rumor_class_dirs = os.listdir(os.path.join(target_path, "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost"))# non-rumor-repost的路径
original_microblog = os.path.join(target_path, "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog")# original-microblog的路径

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
        with open(file_path, 'r',encoding='utf-8') as f:
            rumor_content = f.read()
        rumor_dict = json.loads(rumor_content)
        all_rumor_list.append(rumor_label + "\t" + rumor_dict["text"] + "\n")
        rumor_num += 1
# 解析非谣言数据 analyse non_rumor
for non_rumor_class_dir in non_rumor_class_dirs:
    if not non_rumor_class_dir.endswith('.DS_Store'):
        file_path = os.path.join(original_microblog, non_rumor_class_dir)
        with open(file_path, 'r',encoding='utf-8') as f2:
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
with open(all_data_path, 'w',encoding='utf-8') as f:
    f.seek(0)
    f.truncate()

with open(all_data_path, 'a',encoding='utf-8') as f:
    for data in all_data_list:
        f.write(data)
print('all_data.txt generated!')


# 生成数据字典  dict.txt
# generate the dict.txt

def create_dict(data_path, dict_path, dict_xlsx_path):
    dict_count = {}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # occurences
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            if s in dict_count:
                dict_count[s] += 1
            else:
                dict_count[s] = 1
    
    # 把这些字符及其出现次数保存到本地
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_count))
    
    print("Data dictionary generation completed！", '\t', 'The length of the dictionary：', len(dict_count))

    # turn it into DataFrame
    df = pd.DataFrame(list(dict_count.items()), columns=['character', 'occurrences'])
    
    # turn DataFrame into Excel
    writer = pd.ExcelWriter(dict_xlsx_path)
    df.to_excel(writer, index=False)
    writer._save()
    
    print("Data dictionary.xlsx generation completed！")

data_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/all_data.txt"
dict_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/your_dict_file.txt"
dict_xlsx_path = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/dict.xlsx"
create_dict(data_path, dict_path, dict_xlsx_path)

# Create serialized representation of data and split into training and evaluation sets
# The purpose of this code is to process raw data into data lists for training and  evaluation of text classification models.
# There are TWO file generated：eval_list.txt is for evalution，train_list.txt is foe training。
def create_data_list(data_list_path):
   
    # Open the file containing dictionary data, read it, and parse it into a dictionary.
    with open(os.path.join(data_list_path, 'dict.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    # Open the file containing all the data, read each line of data.
    with open(os.path.join(data_list_path, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    i = 0
    # Create files for writing evaluation data and training data.
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'a', encoding='utf-8') as f_eval, \
            open(os.path.join(data_list_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
        # 遍历所有数据行
        for line in lines:
            # 提取文本标题和标签
            title = line.split('\t')[-1].replace('\n', '')
            lab = line.split('\t')[0]
            t_ids = ""
            # Every 8 lines of data will be used for evaluation, and the rest will be used for training.
            if i % 8 == 0:
                # Convert the title text to the corresponding number in the dictionary and concatenate it into a string.
                for s in title:
                    temp = str(dict_txt[s])
                    t_ids = t_ids + temp + ','
                # 去掉最后一个逗号，然后拼接标签并写入评估文件
                t_ids = t_ids[:-1] + '\t' + lab + '\n'
                f_eval.write(t_ids)
            else:
                # Similarly, convert the title text to the corresponding number in the dictionary and concatenate it into a string.
                for s in title:
                    temp = str(dict_txt[s])
                    t_ids = t_ids + temp + ','
                # 去掉最后一个逗号，然后拼接标签并写入训练文件
                t_ids = t_ids[:-1] + '\t' + lab + '\n'
                f_train.write(t_ids)
            i += 1

    print("Data list generation completed.")

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

