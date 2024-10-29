
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from algorithm.VCCA import VCCA
from sklearn import datasets

import concurrent.futures
import pandas as pd
import os

# 加载数据集
glass = datasets.fetch_openml(data_id=41)
X, y = glass.data, glass.target

# 数据归一化
scaler = MinMaxScaler(feature_range=(0.01, 0.99))
X = scaler.fit_transform(X)

# 创建 LabelEncoder 对象 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 初始化CCA模型
vcca_model = VCCA()
average_acc = 0

# 定义处理单次十折交叉验证的函数
for i in range(10):
    fold_scores = []

    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型
        vcca_model.fit(X_train, y_train)
        # 评估模型
        score = vcca_model.score(X_test, y_test)
        fold_scores.append(score)

    average_score = sum(fold_scores) / len(fold_scores)
    average_acc += average_score
    print(f"正确率{average_score}")


print(f"平均正确率{average_acc/10}")

# # 确保目录存在
# output_dir = '../result'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # 将DataFrame写入Excel文件
# output_file = os.path.join(output_dir, 'glass_VCCA_BaseOnCCA.xlsx')
# df.to_excel(output_file, index=False, engine='openpyxl')