
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import datasets

from algorithm.VCCA import VCCA

import os

def train_VCCA(t, path):
    X, y = t.data, t.target
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    X = scaler.fit_transform(X)

    # 初始化CCA模型
    vcca_model = VCCA()

    for i in range(100):
        average_acc = 0

        # 定义处理单次十折交叉验证的函数
        for i in range(10):
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
            print(f"正确率{average_score}")

        print(f"平均正确率{average_acc / 10}")

    # # 确保目录存在
    # output_dir = '../result'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # # 将DataFrame写入Excel文件
    # output_file = os.path.join(output_dir, path)
    # df.to_excel(output_file, index=False, engine='openpyxl')


if __name__ == '__main__':
    # 加载BCW数据集
    BCW = datasets.load_breast_cancer()


    train_VCCA(BCW)
