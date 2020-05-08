from sklearn.datasets import make_blobs
import numpy as np
import random

'''
    用法:
    kmediod = KMediod(k_num_center=3)
    data = df_train[iv_more_than_point_one]
    data, target = make_blobs(n_samples=1000, n_features=2, centers=1000)
    train_predict = kmediod.fit_transform(train_data)
    test_predict = kmediod.transform(test_data)
'''
class KMediod():
    def __init__(self, k_num_center):
        self.k_num_center = k_num_center
        self.data = None
        self.centroids = None

    def ou_distance(self, x, y):
        # 定义欧式距离的计算
        return np.sqrt(sum(np.square(x - y)))

    def fit(self, data):
        self.data = data.values
        print('初始化', self.k_num_center, '个中心点')
        indexs = list(range(len(self.data)))
        random.shuffle(indexs)  # 随机选择质心
        init_centroids_index = indexs[:self.k_num_center]
        centroids = self.data[init_centroids_index, :]  # 初始中心点
        # 确定种类编号
        levels = list(range(self.k_num_center))
        print('开始迭代')
        sample_target = []
        if_stop = False
        while (not if_stop):
            if_stop = True
            classify_points = [[centroid] for centroid in centroids]
            sample_target = []
            # 遍历数据
            for sample in self.data:
                # 计算距离，由距离该数据最近的核心，确定该点所属类别
                distances = [self.ou_distance(sample, centroid) for centroid in centroids]
                cur_level = np.argmin(distances)
                sample_target.append(cur_level)
                # 统计，方便迭代完成后重新计算中间点
                classify_points[cur_level].append(sample)
            # 重新划分质心
            for i in range(self.k_num_center):  # 几类中分别寻找一个最优点
                distances = [self.ou_distance(point_1, centroids[i]) for point_1 in classify_points[i]]
                now_distances = sum(distances)  # 首先计算出现在中心点和其他所有点的距离总和
                for point in classify_points[i]:
                    distances = [self.ou_distance(point_1, point) for point_1 in classify_points[i]]
                    new_distance = sum(distances)
                    # 计算出该聚簇中各个点与其他所有点的总和，若是有小于当前中心点的距离总和的，中心点去掉
                    if new_distance < now_distances:
                        now_distances = new_distance
                        centroids[i] = point  # 换成该点
                        if_stop = False
        self.centroids = centroids
        print('结束')
        return self

    def fit_transform(self, data):
        self.data = data.values
        print('初始化', self.k_num_center, '个中心点')
        indexs = list(range(len(self.data)))
        random.shuffle(indexs)  # 随机选择质心
        init_centroids_index = indexs[:self.k_num_center]
        centroids = self.data[init_centroids_index, :]  # 初始中心点
        # 确定种类编号
        levels = list(range(self.k_num_center))
        print('开始迭代')
        sample_target = []
        if_stop = False
        while (not if_stop):
            if_stop = True
            classify_points = [[centroid] for centroid in centroids]
            sample_target = []
            # 遍历数据
            for sample in self.data:
                # 计算距离，由距离该数据最近的核心，确定该点所属类别
                distances = [self.ou_distance(sample, centroid) for centroid in centroids]
                cur_level = np.argmin(distances)
                sample_target.append(cur_level)
                # 统计，方便迭代完成后重新计算中间点
                classify_points[cur_level].append(sample)
            # 重新划分质心
            for i in range(self.k_num_center):  # 几类中分别寻找一个最优点
                distances = [self.ou_distance(point_1, centroids[i]) for point_1 in classify_points[i]]
                now_distances = sum(distances)  # 首先计算出现在中心点和其他所有点的距离总和
                for point in classify_points[i]:
                    distances = [self.ou_distance(point_1, point) for point_1 in classify_points[i]]
                    new_distance = sum(distances)
                    # 计算出该聚簇中各个点与其他所有点的总和，若是有小于当前中心点的距离总和的，中心点去掉
                    if new_distance < now_distances:
                        now_distances = new_distance
                        centroids[i] = point  # 换成该点
                        if_stop = False
        self.centroids = centroids
        print('结束')
        return sample_target

    def predict(self, data):
        sample_target = []
        # 遍历数据
        for sample in data:
            # 计算距离，由距离该数据最近的核心，确定该点所属类别
            distances = [self.ou_distance(sample, centroid) for centroid in self.centroids]
            cur_level = np.argmin(distances)
            sample_target.append(cur_level)

        return sample_target