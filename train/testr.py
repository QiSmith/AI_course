
from sklearn import datasets
from sklearn.datasets import fetch_openml, load_digits
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import Bunch
from ucimlrepo import fetch_ucirepo

# 输出glass数据集的特征
def data_describe(zoo, X, y):
    # print(zoo)
    # X = zoo.data
    # y = zoo.target
    # print(X,y)
    # 创建DataFrame来查看特征
    df_X = pd.DataFrame(X, columns=zoo.feature_names)
    df_y = pd.DataFrame(y, columns=['target'])

    # 查看数据集的前几行
    print("数据集前5行:")
    print(df_X.head())
    print(df_y.head())

    # 查看数据集的形状，即行数和列数
    print("\n数据集形状:")
    print(df_X.shape)
    print(df_y.shape)

    # 查看每列的数据类型
    print("\n数据集列的数据类型:")
    print(df_X.dtypes)
    print(df_y.dtypes)

    # 查看每列的缺失值数量
    print("\n每列缺失值数量:")
    print(df_X.isnull().sum())
    print(df_y.isnull().sum())

    # 查看数据集的描述性统计信息
    print("\n数据集描述性统计信息:")
    print(df_X.describe())

    # 查看数据集中的唯一值和它们的计数
    print("\n数据集中的唯一值和它们的计数:")
    print(df_X.nunique())
    print(df_y.nunique())

    # 查看数据集的列名
    print("\n数据集的列名:")
    print(df_X.columns)

    # 如果数据集中包含分类变量，可以查看每个分类变量的值分布
    for column in df_X.select_dtypes(include=['object', 'category']).columns:
        print(f"\n{column}的值分布:")
        print(df_X[column].value_counts())


def car_train():
    # car数据集，需要标签编码
    car = fetch_openml(name='car', version=2)
    X = car.data
    y = car.target

    data_describe(car,X, y)

def fertilizer_train():
    fertility = fetch_ucirepo(id=244)

    X = fertility.data.features
    y = fertility.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    data_describe(fertility,X, y)

def haberman_train():

    # fetch dataset
    haberman_s_survival = fetch_ucirepo(id=43)

    # data (as pandas dataframes)
    X = haberman_s_survival.data.features
    y = haberman_s_survival.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    data_describe(haberman_s_survival,X, y)

def Ionosphere_train():
    # fetch dataset
    ionosphere = fetch_ucirepo(id=52)

    # data (as pandas dataframes)
    X = ionosphere.data.features
    y = ionosphere.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    data_describe(ionosphere,X, y)

def Lymphography_train():
    # fetch dataset
    lymphography = fetch_ucirepo(id=63)

    # data (as pandas dataframes)
    X = lymphography.data.features
    y = lymphography.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    data_describe(lymphography, X, y)

def breast_can_train():
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
    X = np.array([le.fit_transform(X[:, i]) for i in range(X.shape[1])])

    data_describe(breast_cancer, X, y)

if __name__ == '__main__':
    # 获取 Glass Identification 数据集
    # glass = fetch_openml(data_id=41)

    # fertilizer_train()
    # haberman_train()
    # Ionosphere_train()
    # Lymphography_train()
    breast_can_train()