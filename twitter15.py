import pandas as pd
import numpy as np
import re
import nltk
from transformers import BertModel, BertTokenizer
import torch
from collections import Counter

# 下载停用词
from nltk.corpus import stopwords
sw = stopwords.words('english')

# 下载词干提取器
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# 下载词形还原工具
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# 读取数据集 twitter15.txt 到 DataFrame
data = pd.read_csv('C:/Users/38991/Desktop/twitter15_train.txt', sep='\t', header=None)
texts = list(data.iloc[:, 1])

# 指定模型路径
model_path = "C:/Users/38991/Desktop/bert-base-uncased/"

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# 使用 BERT tokenizer 和模型对文本进行编码的函数
def bert_encode(texts):
    input_ids = []
    attention_mask = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=50,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # 修改此处
        )
        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_mask, dim=0)  # 修改此处

# 使用 BERT 对文本进行编码
input_ids, attention_mask = bert_encode(texts)

# 提取 BERT 特征
bert_features = model(input_ids=input_ids, attention_mask=attention_mask)[0]


# 分批处理数据的大小
batch_size = 32

# 将输入数据分成多个批次
num_batches = len(texts) // batch_size + (len(texts) % batch_size != 0)

# 最终的词袋列表
wordbags = list()
for i in range(num_batches):
    # 获取当前批次的文本
    batch_texts = texts[i * batch_size : (i + 1) * batch_size]

    # 使用 BERT 对文本进行编码
    input_ids, attention_mask = bert_encode(batch_texts)

    # 提取 BERT 特征
    with torch.no_grad():
        batch_features = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    # 处理每个样本的特征
    for j in range(len(batch_texts)):
        text = tokenizer.convert_ids_to_tokens(input_ids[j])
        text_clean = list()
        
        for word in text:
            # 过滤一：停用词过滤
            if word not in sw:
                # 过滤二：词形还原
                word_lem = lemmatizer.lemmatize(word)
                # 过滤三：词干提取
                word_stem = stemmer.stem(word_lem)
                text_clean.append(word_stem)
        
        wordbags.append(Counter(text_clean))

df_wb = pd.DataFrame.from_records(wordbags)
df_wb = df_wb.fillna(0).astype(int)

# 添加一个行来计算每个单词的逆文档频率（IDF）值
row_count, column_count = df_wb.shape
df_empty = pd.DataFrame(0, index=[0], columns=df_wb.columns)
df_wb = pd.concat([df_wb, df_empty], ignore_index=True)
df_wb = df_wb.rename(index={row_count: 'IDF'})
for i in range(0, column_count):
    sum = 0
    for j in range(0, row_count):
        if df_wb.iloc[j, i] != 0:
            sum += 1
    df_wb.iloc[row_count, i] = np.log((row_count / sum) + 1)

# 计算每个文本的总词数
df_wb['total_words'] = df_wb.sum(axis=1)

# 计算 TF-IDF
df_wb_prob = df_wb.div(df_wb['total_words'], axis=0).mul(df_wb.iloc[row_count, :], axis=1).round(6)
df_wb_prob = df_wb_prob.iloc[:row_count, :column_count]
# 添加标签到 TF-IDF 矩阵
df_wb_prob['label'] = data.iloc[:, 2]

# 将更新后的 TF-IDF 矩阵保存为 CSV 文件
df_wb_prob.to_csv('tfidf_twitter15_train.csv', index=False)
