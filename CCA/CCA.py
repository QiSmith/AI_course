from scipy.spatial.distance import cdist

import numpy as np

class CCA:
    def __init__(self):
        self.covers = []    # [center, radius, class]
        self.classes = []
        self.num_unknown = [0, 0]
        self.num_known = [0, 0]

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.covers = []

        for cls in self.classes:
            class_samples = X[y == cls]
            dif_class_samples = X[y != cls]

            # 检查是否有未被覆盖的同类样本
            uncovered_class_samples = class_samples.copy()
            print(f'未覆盖样本集:{uncovered_class_samples.shape}')

            while len(uncovered_class_samples) > 0:
                print(f'未覆盖样本集长度:{len(uncovered_class_samples)},样本集类别:{cls}')
                # 随机选择一个未被覆盖的样本作为覆盖中心
                center = self.select_center(uncovered_class_samples)

                radius = self.compute_radius_mid(center, class_samples, dif_class_samples)
                self.covers.append((center, radius, cls))
                # 检查是否有未被覆盖的同类样本
                uncovered_class_samples = [sample for sample in class_samples
                                           if not self.is_covered(sample)]

    # 随机选择覆盖中心
    def select_center(self, class_samples):
        # 随机选择一个样本作为覆盖中心
        indices = np.random.choice(len(class_samples), size=1, replace=False)

        return class_samples[indices[0]]

    def is_covered(self, sample):
        # 检查中心点是否被任何覆盖集覆盖
        for cover in self.covers:
            distance = np.linalg.norm(sample - cover[0])
            if distance <= cover[1]:
                return True
        return False

    # 计算半径，这里简化为最大距离到中心的距离
    def compute_radius_max(self, center, samples):
        distances = cdist( [center], samples, 'euclidean').flatten()
        return np.max(distances)

    def compute_radius_min(self, center, samples):
        # 计算中心点到所有样本的距离
        distances = cdist( [center], samples, 'euclidean').flatten()
        # 取最小距离作为半径
        return np.min(distances)

    def compute_radius_mid(self, center, class_samples, dif_class_samples):
        # 计算异类样本的最小距离
        distances_min = self.compute_radius_min(center, dif_class_samples)
        # 计算同类样本的距离
        distances = cdist([center], class_samples, 'euclidean').flatten()
        distances_max = np.max([x for x in distances
                                if x < distances_min])

        # 如果没有异类样本，使用同类样本的最大距离
        if len(dif_class_samples) <= 0:
            return distances_max
        # 如果只剩下一个点不在覆盖集中，使用该点的模长
        elif len(class_samples) == 1:
            return np.linalg.norm(class_samples)
        else:
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

