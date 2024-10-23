from Cython import ccall
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

from CCA import CCA

iris = datasets.load_iris()
X, y = iris.data, iris.target

# Min-Max归一化
scaler = MinMaxScaler(feature_range=(0.01, 0.99))
# X = scaler.fit_transform(X)

cca_model = CCA()
kf = KFold(n_splits=10, random_state=42, shuffle=True)
# 初始化用于记录每折分数的列表
fold_scores = []

num_known_total = [0, 0]
num_unknown_total = [0, 0]
# 手动实现十折交叉验证的循环
for train_index, test_index in kf.split(X):
    # 分割训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    cca_model.fit(X_train, y_train)
    print(len(cca_model.covers))
    # 在测试集上评估模型
    score = cca_model.score(X_test, y_test)
    fold_scores.append(score)

    num_known_total[0] += cca_model.num_known[0]
    num_known_total[1] += cca_model.num_known[1]
    num_unknown_total[0] += cca_model.num_unknown[0]
    num_unknown_total[1] += cca_model.num_unknown[1]

# 输出每一折的分数和平均分数
for i, score in enumerate(fold_scores, 1):
    print(f"Fold {i} accuracy: {score:.4f}")

print(f"Average accuracy: {sum(fold_scores) / len(fold_scores):.4f}")

num_known_total = [x / 10 for x in num_known_total]
num_unknown_total = [x / 10 for x in num_unknown_total]


print(f"可识别样本数: {num_known_total[0]},可识别样本的正确数: {num_known_total[1]},可识别样本正确率: {num_known_total[1]/num_known_total[0]}")
# print(f"不可识别样本数: {num_unknown_total[0]},不可识别样本的正确数: {num_unknown_total[1]},不可识别样本正确率: {num_unknown_total[1]/num_unknown_total[0]}")

# cca_model = CCA()
# kf = KFold(n_splits=10, random_state=42, shuffle=True)
#
# scores = cross_val_score(cca_model, X, y, cv=kf)
#
# # 输出每一折的准确率和平均准确率
# for i, score in enumerate(scores):
#     print(f"Fold {i+1} accuracy: {score:.4f}")
#
# print(f"Average accuracy: {scores.mean():.4f}")