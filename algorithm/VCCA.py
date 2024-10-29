
import numpy as np
from algorithm.CCA import CCA

class VCCA:
    def __init__(self, num_models=15):
        self.num_models = num_models
        self.cca_models = []

    def fit(self, X, y):
        self.cca_models = [CCA() for _ in range(self.num_models)]
        for model in self.cca_models:
            model.fit(X, y)

    def predict(self, X):
        predictions = []
        for x in X:
            model_preds = [model.predict([x], None, flag=True)[0] for model in self.cca_models]
            # 使用投票机制确定最终预测
            prediction = np.bincount(model_preds).argmax()
            predictions.append(prediction)
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)