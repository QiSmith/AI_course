
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn import datasets

from algorithm.VCCA import VCCA
from algorithm.CCA import CCA

# 加载数据集
BCW = datasets.load_breast_cancer()
X, y = BCW.data, BCW.target

# 数据归一化
scaler = MinMaxScaler(feature_range=(0.01, 0.99))
X = scaler.fit_transform(X)

# 创建 LabelEncoder 对象 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 初始化CCA模型
vcca_model = VCCA()
cca_model = CCA()
average_acc = 0

# 定义处理单次十折交叉验证的函数
for i in range(10):
    fold_scores = []

    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # # VCCA训练、评估模型
        # vcca_model.fit(X_train, y_train)
        # score = vcca_model.score(X_test, y_test)

        # CCA训练、评估模型
        cca_model.fit(X_train, y_train)
        score = cca_model.score(X_test, y_test)

        fold_scores.append(score)

    average_score = sum(fold_scores) / len(fold_scores)
    average_acc += average_score
    print(f"正确率{average_score}")

print(f"平均正确率{average_acc / 10}")
