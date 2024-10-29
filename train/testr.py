
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

# 输出glass数据集的特征
def data_describe(zoo):
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
    # for column in df_X.select_dtypes(include=['object', 'category']).columns:
    #     print(f"\n{column}的值分布:")
    #     print(df_X[column].value_counts())

if __name__ == '__main__':
    # 获取 Glass Identification 数据集
    # glass = fetch_openml(data_id=41)
    # 加载数据集iris
    zoo = fetch_openml(name='zoo', version=1)
    data_describe(zoo)