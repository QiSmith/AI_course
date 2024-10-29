from pyasn1_modules.rfc5990 import nistAlgorithm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn import datasets

import concurrent.futures
import pandas as pd
import numpy as np
import os

from algorithm.CCA import CCA

def train_cca(data, path, num_train):

    X, y = data.data, data.target

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    X = scaler.fit_transform(X)

    # 初始化CCA模型
    cca_model = CCA()

    # 初始化用于记录每折分数的列表
    df = pd.DataFrame(columns=[
        '平均覆盖集个数', '可识别的样本数', '可识别样本的正确数', '可识别样本的正确率',
        '不可识别的样本数', '不可识别样本的正确数', '不可识别样本的正确率', '总正确率',
    ])

    # 定义处理单次十折交叉验证的函数
    for i in range(num_train):
        num_known_total = [0, 0]
        num_unknown_total = [0, 0]
        num_covers = 0
        fold_scores = []

        kf = KFold(n_splits=10, random_state=42, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 训练模型
            cca_model.fit(X_train, y_train)
            # 评估模型
            score = cca_model.score(X_test, y_test)
            fold_scores.append(score)

            num_covers += len(cca_model.covers)
            num_known_total[0] += cca_model.num_known[0]
            num_known_total[1] += cca_model.num_known[1]
            num_unknown_total[0] += cca_model.num_unknown[0]
            num_unknown_total[1] += cca_model.num_unknown[1]

        # 计算平均值
        num_known_total = [x / 10 for x in num_known_total]
        num_unknown_total = [x / 10 for x in num_unknown_total]
        num_covers /= 10
        average_score = sum(fold_scores) / len(fold_scores)

        # 创建结果字典
        result = {
            '平均覆盖集个数': num_covers,
            '可识别的样本数': num_known_total[0],
            '可识别样本的正确数': num_known_total[1],
            '可识别样本的正确率': num_known_total[1] / num_known_total[0],
            '不可识别的样本数': num_unknown_total[0],
            '不可识别样本的正确数': num_unknown_total[1],
            '不可识别样本的正确率': num_unknown_total[1] / num_unknown_total[0],
            '总正确率': average_score,
        }
        return result


    # 确保目录存在
    output_dir = '../result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将DataFrame写入Excel文件
    output_file = os.path.join(output_dir, path)
    df.to_excel(output_file, index=False, engine='openpyxl')

# 定义一个函数来转换数据集中的布尔型和其他类别型特征
def encode(X):
    # 创建一个空列表来存储转换后的数据
    encoded_X = []
    # 遍历每一列
    for i in range(X.shape[1]):
        column = X[:, i]
        # 检查是否为布尔型特征
        if column.dtype == 'object' and set(column).issubset({'true', 'false'}):
            # 使用条件表达式转换布尔型特征
            encoded_column = [1 if val == 'true' else 0 for val in column]
        elif column.dtype == 'object':
            # 使用LabelEncoder转换类别型特征
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded_column = le.fit_transform(column)
        else:
            # 其他数值型特征保持不变
            encoded_column = column
        encoded_X.append(encoded_column)
    # 将转换后的列组合成一个NumPy数组
    return np.array(encoded_X).T

if __name__ == '__main__':
    path = 'iris_Mid_MinDistWithCenter.xlsx'

    # 加载数据集iris
    # iris = datasets.load_iris()

    # 加载数据集wine
    # wine = datasets.load_wine()

    # 加载数据集zoo
    zoo = datasets.fetch_openml(name='zoo', version=1)
    # zoo.data = encode(zoo.data)

    data = zoo
    train_cca(data, path, 100)