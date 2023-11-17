import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# 加载TF-IDF矩阵和标签数据
df = pd.read_csv('C:/Users/38991/Desktop/tfidf_twitter15_train.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# 对标签进行独热编码
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
y_encoded = label_encoder.fit_transform(y)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# 将输入数据调整为三维
X = np.expand_dims(X, axis=2)

# 定义五折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []
recall_scores = []
precision_scores = []

# 构建CNN模型
model = Sequential() 
# 添加第一个卷积层，包含64个过滤器，卷积核大小为3，激活函数为ReLU，输入形状为(X.shape[1], X.shape[2])
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
# 添加全局最大池化层
model.add(GlobalMaxPooling1D())
# 添加一个具有128个神经元和ReLU激活函数的全连接层
model.add(Dense(128, activation='relu'))
# 添加Dropout层，以减少过拟合，丢弃比例为0.5
model.add(Dropout(0.5))
# 添加输出层，使用softmax激活函数将输出映射为类别概率
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
# 编译模型，使用Adam优化器、分类交叉熵损失函数和准确率评估指标
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# 打印模型概述信息
model.summary()

# 执行五折交叉验证
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_onehot[train_index], y_onehot[test_index]


    # 训练模型
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # 在测试集上进行预测
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 将预测结果转换为独热编码标签
    y_pred_orig = onehot_encoder.inverse_transform(np.eye(len(label_encoder.classes_))[y_pred])

    # 将真实标签转换为独热编码标签
    y_test_orig = onehot_encoder.inverse_transform(y_test)

    # 计算评估指标
    f1 = f1_score(y_test_orig, y_pred_orig, average='weighted')
    recall = recall_score(y_test_orig, y_pred_orig, average='weighted')
    precision = precision_score(y_test_orig, y_pred_orig, average='weighted', zero_division=1)

# 计算平均评估指标值
avg_f1 = np.mean(f1)
avg_recall = np.mean(recall)
avg_precision = np.mean(precision)
 
# 绘制柱状图
metrics = ['F1 Score', 'Recall', 'Precision']
scores = [avg_f1, avg_recall, avg_precision]
plt.bar(metrics, scores)
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Evaluation Metrics')
plt.show()

# 在测试集上进行预测
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# 将预测结果转换为独热编码标签
y_pred_orig = label_encoder.inverse_transform(y_pred)

# 计算混淆矩阵
cm = confusion_matrix(y[test_index], y_pred_orig)

# 绘制混淆矩阵图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



