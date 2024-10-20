from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# import iris dataset 数据 和 标签
iris = load_iris()
X,y = iris.data, iris.target

# print the number of sample and the number of features
# print(X.shape)
# print(y.shape)

# df = pd.DataFrame(X,columns=iris.feature_names)
# df['target'] = y

# # 查看数据集
# print(df)
#
# # 查看数据集的描述性统计信息
# print(df.describe())
#
# # 查看数据集的信息，包括每列的数据类型和非空值数量
# print(df.info())

# # 导出为xlsx文件
# iris_csv_path = 'dataset/iris/iris.xlsx'
# df.to_excel(iris_csv_path,index=False)

# # 检查空值
# print(df.isnull().sum())
#
# # 删除含有缺失值的行
# df.dropna(inplace=True)
#
# # 用平均值填充缺失值
# df.fillna(df.mean(), inplace=True)
#
# # 查看数据类型
# print(df.dtypes)
# # 如果需要，转换数据类型
# # df['column_name'] = df['column_name'].astype('float')

scaler = MinMaxScaler()
scaler_data = scaler.fit_transform(X)
