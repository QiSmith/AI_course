
from sklearn.datasets import fetch_openml, load_iris
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from ucimlrepo import fetch_ucirepo

from algorithm.VCCA import VCCA

import pandas as pd
import numpy as np
import time
import os


def train_VCCA(X, y, path, num_epoch):
    # 记录开始时间
    start_time = time.time()

    # X, y = t.data, t.target # sklearn获取的是 bunch对象，通过这种方式得到属性和标签
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    X = scaler.fit_transform(X)

    # 初始化用于记录每折分数的列表
    df = pd.DataFrame(columns=[
        '正确率','平均正确率','标准差'
    ])

    average_acc = 0
    # 定义处理单次十折交叉验证的函数
    for j in range(num_epoch):
        fold_scores = []
        # 初始化VCCA模型
        vcca_model = VCCA()

        kf = KFold(n_splits=10, random_state=42, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # VCCA训练、评估模型
            vcca_model.fit(X_train, y_train)
            score = vcca_model.score(X_test, y_test)

            fold_scores.append(score)

        average_score = sum(fold_scores) / len(fold_scores)
        average_acc += average_score*100
        print(f"num_epoch:{j}")

        result={
            '正确率':average_score*100,
        }
        # 将字典转换为DataFrame
        result_df = pd.DataFrame([result])

        # 追加到原始DataFrame中
        df = df.append(result_df, ignore_index=True)

    std_dev = df['正确率'].std()
    result={
        '平均正确率':average_acc/num_epoch,
        '标准差':std_dev/100
    }
    result_df = pd.DataFrame([result])
    df = df.append(result_df, ignore_index=True)

    # 确保目录存在
    output_dir = '../n-delete-result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将DataFrame写入Excel文件
    output_file = os.path.join(output_dir, path)
    df.to_excel(output_file, index=False, engine='openpyxl')

    # 记录结束时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间：{execution_time} 秒")


def one_hot_encode_non_numeric(df):
    """
    对DataFrame中所有非数值列进行独热编码。
    :param
    df (DataFrame): 包含要编码列的DataFrame。
    :return
    DataFrame: 包含所有非数值列独热编码的新DataFrame。
    """

    # 选择非数值列
    cols_to_encode = df.select_dtypes(exclude=['int64', 'float64']).columns

    # 对指定列进行独热编码
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    return df_encoded

def iris_train():
    path = 'Iris_VCCA.xlsx'
    # iris 数据集
    iris = load_iris()
    X = iris.data   # numpy.ndarray
    y = iris.target

    train_VCCA(X, y, path, 20)

def fertilizer_train():
    path = 'Fertilizer_VCCA.xlsx'
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
    train_VCCA(X, y, path, 20)

def haberman_train():
    path = 'Haberman_VCCA.xlsx'
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

    train_VCCA(X, y, path, 20)

def Ionosphere_train():
    path = 'Ionosphere_VCCA.xlsx'

    # fetch dataset
    ionosphere = fetch_ucirepo(id=52)


    # data (as pandas dataframes)
    X = ionosphere.data.features
    y = ionosphere.data.targets
    # 第二列数据只有一个值
    X = X.drop(X.columns[1],axis=1)

    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # 初始化 LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    train_VCCA(X, y, path, 20)

def Lymphography_train():
    path = 'Lymphography_VCCA.xlsx'
    # fetch dataset
    lymphography = fetch_ucirepo(id=63)

    # data (as pandas dataframes)
    X = lymphography.data.features
    y = lymphography.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # 删除第19列，因为全是NaN
    X = np.delete(X, 18, axis=1)

    train_VCCA(X, y, path, 20)

def breast_can_train():
    path = 'Breast_can_VCCA.xlsx'
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

    train_VCCA(encoded_X, y, path, 20)

# 存在nan
def ilpd_train():
    path = 'ILPD_VCCA.xlsx'
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

    train_VCCA(X, y, path, 20)

def segmentation_train():
    path = 'Segmentation_VCCA.xlsx'
    # fetch dataset
    image_segmentation = fetch_ucirepo(id=50)

    # data (as pandas dataframes)
    X = image_segmentation.data.features
    y = image_segmentation.data.targets

    X = X.to_numpy()
    y = y.to_numpy().flatten()

    # 初始化 LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    train_VCCA(X, y, path, 5)

def balance_train():
    path = 'balance_VCCA.xlsx'
    # fetch dataset
    balance_scale = fetch_ucirepo(id=12)

    # data (as pandas dataframes)
    X = balance_scale.data.features
    y = balance_scale.data.targets

    X = one_hot_encode_non_numeric(X)

    X = X.to_numpy()
    y = y.to_numpy().flatten()

    le = LabelEncoder()
    y = le.fit_transform(y)

    train_VCCA(X, y, path, 10)

if __name__ == '__main__':
    # 属性编码用 X = one_hot_encode_non_numeric(X) 标签编码用
    # 初始化 LabelEncoder
    # le = LabelEncoder()
    # y = le.fit_transform(y)

    iris_train()
    # fertilizer_train()
    haberman_train()
    # Ionosphere_train()
    # Lymphography_train()
    # breast_can_train()
    # ilpd_train()
    # segmentation_train()
    # balance_train()