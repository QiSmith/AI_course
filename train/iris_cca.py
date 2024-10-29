
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import datasets

import concurrent.futures
import pandas as pd
import os

from algorithm.CCA import CCA

# 加载数据集iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

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
def process_single_fold(index):
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


# 使用多线程执行100次十折交叉验证
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    futures = [executor.submit(process_single_fold, i) for i in range(100)]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        df = df.append(result, ignore_index=True)

# 确保目录存在
output_dir = '../result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将DataFrame写入Excel文件
output_file = os.path.join(output_dir, 'iris_Mid_MinDistWithCenter.xlsx')
# output_file = os.path.join(output_dir, 'Glass_CCA_Mid_MinDistWithCenter.xlsx')
df.to_excel(output_file, index=False, engine='openpyxl')