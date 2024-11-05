
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn import datasets

import pandas as pd
import numpy as np
import time
import os

from ucimlrepo import fetch_ucirepo

from algorithm.CCA import CCA

def train_cca(X, y, path, num_train):

    # X, y = data.data, data.target

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

        print(f"num_epoch：{i}")
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

        # 将字典转换为DataFrame
        result_df = pd.DataFrame([result])

        # 追加到原始DataFrame中
        df = df.append(result_df, ignore_index=True)
    std_dev = df['总正确率'].std()
    # 确保目录存在
    output_dir = '../result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将DataFrame写入Excel文件
    output_file = os.path.join(output_dir, path)
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(std_dev)

def iris_train():
    path = 'Iris_CCA.xlsx'
    # iris 数据集
    iris = datasets.load_iris()
    X = iris.data   # numpy.ndarray
    y = iris.target

    train_cca(X, y, path, 10)


def fertilizer_train():
    path = 'Fertilizer_CCA.xlsx'
    # fetch dataset
    fertility = fetch_ucirepo(id=244)

    # data (as pandas dataframes)
    X = fertility.data.features
    y = fertility.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # 初始化 LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # print(y.shape,y)
    train_cca(X, y, path, 20)

def breast_can_train():
    path = 'Breast_can_CCA.xlsx'
    # fetch dataset
    breast_cancer = fetch_ucirepo(id=14)

    # data (as pandas dataframes)
    X = breast_cancer.data.features
    y = breast_cancer.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # 初始化 LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    # 对每一列进行标签编码，同时保持数组形状不变
    encoded_X = np.empty_like(X)  # 创建一个与X形状相同的空数组
    for i in range(X.shape[1]):  # 遍历每一列
        encoded_X[:, i] = le.fit_transform(X[:, i])  # 对每一列进行标签编码

    train_cca(encoded_X, y, path, 10)

def haberman_train():
    path = 'Haberman_CCA.xlsx'
    # fetch dataset
    haberman_s_survival = fetch_ucirepo(id=43)

    # data (as pandas dataframes)
    X = haberman_s_survival.data.features
    y = haberman_s_survival.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # 初始化 LabelEncoder
    # le = LabelEncoder()
    # y = le.fit_transform(y)

    train_cca(X, y, path, 20)

def Ionosphere_train():
    path = 'Ionosphere_CCA.xlsx'

    # fetch dataset
    ionosphere = fetch_ucirepo(id=52)

    # data (as pandas dataframes)
    X = ionosphere.data.features
    y = ionosphere.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # 初始化 LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    train_cca(X, y, path, 20)

def Lymphography_train():
    path = 'Lymphography_CCA.xlsx'
    # fetch dataset
    lymphography = fetch_ucirepo(id=63)

    # data (as pandas dataframes)
    X = lymphography.data.features
    y = lymphography.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # 删除第19列，因为全是NaN
    X = np.delete(X, 18, axis=1)

    train_cca(X, y, path, 20)


if __name__ == '__main__':
    # fertilizer_train()
    # breast_can_train()
    # haberman_train()
    # iris_train()
    # Ionosphere_train()
    Lymphography_train()