
from sklearn.datasets import fetch_openml, load_iris
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from ucimlrepo import fetch_ucirepo
from sklearn.utils import Bunch
from sklearn import datasets


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
    # y = y.values

    # 初始化CCA模型
    vcca_model = VCCA()

    # 初始化用于记录每折分数的列表
    df = pd.DataFrame(columns=[
        '正确率',
    ])

    for i in range(num_epoch):
        average_acc = 0

        # 定义处理单次十折交叉验证的函数
        for j in range(15):
            fold_scores = []

            kf = KFold(n_splits=10, random_state=42, shuffle=True)

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # VCCA训练、评估模型
                vcca_model.fit(X_train, y_train)
                score = vcca_model.score(X_test, y_test)

                fold_scores.append(score)

            average_score = sum(fold_scores) / len(fold_scores)
            average_acc += average_score
            # print(average_acc)
        print(f"num_epoch:{i}")

        result={
            '正确率':average_acc/15,
        }
        # 将字典转换为DataFrame
        result_df = pd.DataFrame([result])

        # 追加到原始DataFrame中
        df = df.append(result_df, ignore_index=True)
        # print(df)


    std_dev = df['正确率'].std()

    # 确保目录存在
    output_dir = '../result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将DataFrame写入Excel文件
    output_file = os.path.join(output_dir, path)
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"标准差:{std_dev}")

    # 记录结束时间
    end_time = time.time()

    # 计算执行时间
    execution_time = end_time - start_time
    print(f"程序执行时间：{execution_time} 秒")


def label_encode_bunch(bunch):
    # 将数据转换为 DataFrame
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)

    # 识别非数值列（即分类数据）
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # 初始化 LabelEncoder
    le = LabelEncoder()

    # 对每个分类列进行标签编码
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # 对目标变量进行标签编码
    # 假设目标变量是 bunch.target，且它是一维数组
    if isinstance(bunch.target, np.ndarray):
        unique_classes = np.unique(bunch.target)
        class_dict = {cls: idx for idx, cls in enumerate(unique_classes)}
        bunch.target = np.array([class_dict[cls] for cls in bunch.target])
    elif isinstance(bunch.target, pd.Series):
        unique_classes = bunch.target.unique()
        class_dict = {cls: idx for idx, cls in enumerate(unique_classes)}
        bunch.target = bunch.target.map(class_dict).values

    # 提取特征和目标变量
    X_encoded = df.values
    y_encoded = bunch.target

    # 创建新的 Bunch 对象
    new_bunch = Bunch(
        data=X_encoded,
        target=y_encoded,
        feature_names=df.columns.tolist(),
        target_names=[str(cls) for cls in sorted(set(y_encoded))],  # 更新目标名称为编码后的类别标签
        DESCR=bunch.DESCR
    )

    return new_bunch

def iris_train():
    path = 'Iris_VCCA.xlsx'
    # iris 数据集
    iris = load_iris()
    X = iris.data   # numpy.ndarray
    y = iris.target

    train_VCCA(X, y, path, 1)

def car_train():
    path = 'Car_VCCA.xlsx'
    # car数据集，需要标签编码
    car = fetch_openml(name='car', version=2)
    car_data = label_encode_bunch(car)
    X = car_data.data
    y = car_data.target

    train_VCCA(X, y, path, 1)

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

    train_VCCA(encoded_X, y, path, 10)

if __name__ == '__main__':
    # fertilizer_train()
    # iris_train()
    # car_train()
    # haberman_train()
    # Ionosphere_train()
    # Lymphography_train()
    breast_can_train()