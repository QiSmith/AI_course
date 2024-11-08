
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
import seaborn as sns

import pandas as pd

# 输出glass数据集的特征
def data_describe(X, y):
    df_X = pd.DataFrame(X)
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

    # # 如果数据集中包含分类变量，可以查看每个分类变量的值分布
    # for column in df_X.select_dtypes(include=['object', 'category']).columns:
    #     print(f"\n{column}的值分布:")
    #     print(df_X[column].value_counts())

    # # 检查异常值
    # print("\n异常值检查（箱线图）:")
    # plt.figure(figsize=(10, 8))
    # sns.boxplot(data=df_X)
    # plt.xticks(rotation=90)
    # plt.show()
    #
    # # 数据分布
    # print("\n数值型数据分布（直方图）:")
    # plt.figure(figsize=(10, 8))
    # sns.histplot(df_X, kde=True)
    # plt.xticks(rotation=90)
    # plt.show()
    #
    # # 相关性分析
    # print("\n变量相关性（热图）:")
    # plt.figure(figsize=(10, 8))
    # corr = df_X.corr()
    # sns.heatmap(corr, annot=True, cmap='coolwarm')
    # plt.show()

    # 数据一致性检查
    print("\n数据一致性检查:")
    for column in df_X.select_dtypes(include=['object', 'category']).columns:
        print(f"\n{column}的合法类别值检查:")
        print(df_X[column].apply(lambda x: x in df_X[column].unique()))

def one_hot_encode_non_numeric(df, flag=False):
    """
    对DataFrame中所有非数值列进行独热编码。
    :param
    df (DataFrame): 包含要编码列的DataFrame。
    :return
    DataFrame: 包含所有非数值列独热编码的新DataFrame。
    """
    if not flag:
        # 选择非数值列
        cols_to_encode = df.select_dtypes(exclude=['int64', 'float64']).columns
    else:
        # 如果flag=True，则选择所有列
        cols_to_encode = df.columns
    # 对指定列进行独热编码
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    return df_encoded

def fertilizer_train():
    fertility = fetch_ucirepo(id=244)

    X = fertility.data.features
    y = fertility.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    data_describe(X, y)

def haberman_train():

    # fetch dataset
    haberman_s_survival = fetch_ucirepo(id=43)

    # data (as pandas dataframes)
    X = haberman_s_survival.data.features
    y = haberman_s_survival.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    data_describe(X, y)

def Ionosphere_train():
    # fetch dataset
    ionosphere = fetch_ucirepo(id=52)

    # data (as pandas dataframes)
    X = ionosphere.data.features
    y = ionosphere.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    le = LabelEncoder()
    y = le.fit_transform(y)

    data_describe(X, y)

def Lymphography_train():
    # fetch dataset
    lymphography = fetch_ucirepo(id=63)

    # data (as pandas dataframes)
    X = lymphography.data.features
    y = lymphography.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    data_describe(X, y)

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

    data_describe(X, y)

def ilpd_train():
    # fetch dataset
    ilpd_indian_liver_patient_dataset = fetch_ucirepo(id=225)

    # data (as pandas dataframes)
    X = ilpd_indian_liver_patient_dataset.data.features
    y = ilpd_indian_liver_patient_dataset.data.targets

    # 检查是否存在NaN值
    if X.isnull().any().any() or y.isnull().any():
        # 删除包含NaN的行
        X = X.dropna()
        y = y.loc[X.index]  # 确保y与X的行对应

        # 如果y是一个DataFrame，也需要删除NaN值
        if y.isnull().any().any():
            y = y.dropna()

    X = one_hot_encode_non_numeric(X)
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # data_describe(X, y)

def segmentation_train():
    # fetch dataset
    image_segmentation = fetch_ucirepo(id=50)

    # data (as pandas dataframes)
    X = image_segmentation.data.features
    y = image_segmentation.data.targets

    X = X.to_numpy()
    y = y.to_numpy().flatten()

    data_describe(X, y)

def balance_train():
    # fetch dataset
    balance_scale = fetch_ucirepo(id=12)

    # data (as pandas dataframes)
    X = balance_scale.data.features
    y = balance_scale.data.targets

    X = one_hot_encode_non_numeric(X, flag=True)

    X = X.to_numpy()
    y = y.to_numpy().flatten()

    le = LabelEncoder()
    y = le.fit_transform(y)

    data_describe(X, y)

if __name__ == '__main__':
    # fertilizer_train()
    # haberman_train()
    Ionosphere_train()
    # Lymphography_train()
    # breast_can_train()
    # ilpd_train()
    # segmentation_train()
    # balance_train()