from scipy.spatial.distance import cdist

import numpy as np

class CCA:
    def __init__(self,radius=None):
        self.radius = radius
        self.covers = []
        self.X_train = []
        self.y_train = []
        self.classes = []
        self.num_unknown = [0, 0]
        self.num_known = [0, 0]

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y) # 标签
        self.covers = []    # center, radius, class

        for cls in self.classes:
            class_mask = (y == cls)
            class_samples = X[class_mask]   # 选择出所有cls类别的样本
            dif_class_samples = X[y != cls]   # 选择出所有非cls类别的样本
            if len(class_samples) > 0:
                cover_centers = self.select_centers(class_samples)
                for center in cover_centers:
                    radius = self.compute_radius_max(center,class_samples)
                    # radius = self.compute_radius_max(center,class_samples)
                    # radius = self.compute_radius_min(center,class_samples)
                    self.covers.append((center, radius, cls))

    # 随机选择覆盖中心
    def select_centers(self, class_samples):
        indices = np.random.choice(class_samples.shape[0], size=5, replace=False)
        return class_samples[indices]

    # 计算半径，这里简化为最大距离到中心的距离
    def compute_radius_max(self, center, class_samples):
        distances = cdist(class_samples, [center], 'euclidean').flatten()
        return np.max(distances)

    def compute_radius_min(self, center, class_samples):
        # 计算中心点到同类样本的最小距离
        distances = cdist(class_samples, [center], 'euclidean').flatten()
        # 最小距离作为半径
        return np.min(distances)

    def compute_radius_mid(self, center, class_samples):
        distances_max = self.compute_radius_max(center, class_samples)
        distances_min = self.compute_radius_min(center, class_samples)

        return (distances_max + distances_min) / 2

    def predict(self, X, y_true):
        predictions = []
        for x, test_label in zip(X, y_true):
            # 存储每个覆盖集的距离和对应的类别
            distances_and_classes = []  # distance, class
            covered = False

            for cover in self.covers:
                # 使用欧氏距离计算样本到覆盖集中心点的距离
                distance = np.linalg.norm(x - cover[0])
                # 检查是否在覆盖集内部
                if distance <= cover[1]:
                    distances_and_classes.append((distance, cover[2]))
                    covered = True

            if not covered:
                # 如果样本没有落入任何覆盖集，选择欧氏距离最近的覆盖集
                nearest_cover_index = np.argmin([np.linalg.norm(x - cover[0]) for cover in self.covers])    # 距中心距离
                # nearest_cover_index = np.argmin([np.linalg.norm(x - cover[0]) - cover[1] for cover in self.covers])    # 距边界距离
                nearest_cover = self.covers[nearest_cover_index]
                predictions.append(nearest_cover[2])

                self.num_unknown[0] += 1
                if nearest_cover[2] == test_label:
                    self.num_unknown[1] += 1
            else:
                if len(distances_and_classes) == 1:
                    # 如果只有一个覆盖集，直接选择该覆盖集的类别
                    predictions.append(distances_and_classes[0][1])
                    self.num_known[0] += 1
                    if distances_and_classes[0][1] == test_label:
                        self.num_known[1] += 1
                else:
                    # # 使用投票方法
                    # prediction = self.vote_predictions(distances_and_classes)
                    # predictions.append(prediction)

                    prediction = self.dist_center(X)
                    predictions.append(prediction)
                    self.num_known[0] += 1
                    if prediction == test_label:
                        self.num_known[1] += 1
        return np.array(predictions)

    # 距中心最近
    def dist_center(self, x):
        nearest_cover_index = np.argmin([np.linalg.norm(x - cover[0]) for cover in self.covers])
        nearest_cover = self.covers[nearest_cover_index]
        return nearest_cover[2]

    def vote_predictions(self, distances_and_classes):
        class_votes = {}
        for dist,cls in distances_and_classes:
            if cls not in class_votes:
                class_votes[cls] = 0
            class_votes[cls] += 1  # 简单的投票机制

        # 选择得票最多的类别
        prediction = max(class_votes, key=class_votes.get)
        return prediction

    def gravitational_predictions(self, distances_and_classes):
        total_mass = sum(1 / distance for distance, _ in distances_and_classes)
        class_masses = {}
        for distance, cls in distances_and_classes:
            if cls not in class_masses:
                class_masses[cls] = 0
            class_masses[cls] += (1 / distance) / total_mass

        # 选择质量最大的类别
        prediction = max(class_masses, key=class_masses.get)
        return prediction

    def score(self, X, y):
        predictions = self.predict(X, y)
        return np.mean(predictions == y)

