import numpy as np
from random import random
from kmeans import load_data, kmeans, distance, save_result

FLOAT_MAX = 1e100 # 设置一个较大的值作为初始化的最小的距离

def nearest(point, cluster_centers):
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])   #计算当前点与已经初始化的每个点的距离，并且得到最小值，即假设该点与初始化的第二个簇中心最短
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist

def get_centroids(points, k):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k , n)))
    # 1、随机选择一个样本点为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j, ])
            break
    return cluster_centers

'''
关于kmeans++初始化种子点的方法：
1. 首先在所有的n个点点里面初始化一个点作为第一个种子点
2. 计算其他n个点到该点的距离，并且计入d[n]这个数组里面
3. 接下来的思想：假设分为距离依次为 10 5 20 25 0 30 10，那么可以发现，应该将6个点作为下一个种子点，但是这种方法需要比较距离的大小，所以我们使用概率的方式
  假设这是几个段的长度，随机扔一个点，那么落在30这个段的概率更大，但是也不一定，所以最终将sumd[n]加起来，乘以一个随机值得到r标准，最后再从头累加10+5+20+25+...
  一旦超过了这个值，即判断该点为下一个随机种子点。
4. 得到两个种子点后，则是计算剩下的点到这两个点中的最小距离存为d[n]，循环操作，直到找到所有的种子点。

kmeans++得到种子点后，按照kmeans的方法继续聚类，因此最大的区别就是采用的概率的方式获得初始化的种子点。改掉了kmeans方法严重受随机初始种子点的影响。
'''


if __name__ == "__main__":
    k = 4#聚类中心的个数
    file_path = "data.txt"
    # 1、导入数据
    print ("---------- 1.load data ------------")
    data = load_data(file_path)
    # 2、KMeans++的聚类中心初始化方法
    print("---------- 2.K-Means++ generate centers ------------")
    centroids = get_centroids(data, k)
    # 3、聚类计算
    print("---------- 3.kmeans ------------")
    subCenter = kmeans(data, k, centroids)
    # 4、保存所属的类别文件
    print("---------- 4.save subCenter ------------")
    save_result("sub_pp", subCenter)
    # 5、保存聚类中心
    print("---------- 5.save centroids ------------")
    save_result("center_pp", centroids)
