import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt


interval_km = 0.001  # 聚族内车辆最大的间距
min_car_sample_count = 15  # 聚族最小车辆样本数


def dbscan(input_file):
    data = pd.read_excel(input_file)
    coords = data.values

    kms_per_radian = 6371.0086

    epsilon = interval_km / kms_per_radian

    db = DBSCAN(
        eps=epsilon, min_samples=150, algorithm="ball_tree", metric="haversine"
    ).fit(np.radians(coords))
    cluster_labels = db.labels_

    num_clusters = len(set(cluster_labels) - set([-1]))
    print(
        "Clustered "
        + str(len(coords))
        + " points to "
        + str(num_clusters)
        + " clusters"
    )

    plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels)

    plt.show()
    plt.savefig("test22.png")

    for n in range(num_clusters):
        one_cluster = coords[cluster_labels == n]
        print(one_cluster[:1])


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(
        cluster, key=lambda point: great_circle(point, centroid).miles
    )
    return tuple(centermost_point)


def main():
    path = "test2.xlsx"
    dbscan(path)


if __name__ == "__main__":
    main()