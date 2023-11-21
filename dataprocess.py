# -*- coding: utf-8 -*-

'''
@article{song2018ced,
# title={CED: Credible Early Detection of Social Media Rumors},
#  author={Song, Changhe and Tu, Cunchao and Yang, Cheng and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:1811.04175},
  year={2018}
'''

import os
import json
import random

# 分别为谣言数据、非谣言数据、全部数据的文件路径
# Define paths for rumor 、non-rumor and original_data datasets
rumor_class_dirs = os.listdir("C:/users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost")
# 这里填写自己项目所在的数据集路径
non_rumor_class_dirs = os.listdir(
    "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost")
original_microblog = "C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog"

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

with open(all_data_path, 'a', encoding="utf-8") as f:
    for data in all_data_list:  # 按行写入，一行一个样本
        f.write(data)

test_data_list = all_data_list[:int(0.1*len(all_data_list))]
train_data_list = all_data_list[len(test_data_list):]
print(len(test_data_list))
print(len(train_data_list))

with open("C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/train.txt", 'a', encoding="utf-8") as f:
    for data in train_data_list:  # 按行写入，一行一个样本
        f.write(data)

with open("C:/Users/杨思博/Desktop/Chinese_Rumor_Dataset-master/CED_Dataset/test.txt", 'a', encoding="utf-8") as f:
    for data in test_data_list:  # 按行写入，一行一个样本
        f.write(data)