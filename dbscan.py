import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics

def dbscan(input_file):
    data = pd.read_excel(input_file)
    coords = data.loc[:,['lon','lat']].values

    coords = [[int(item) / 1000000 for item in coord if item and item != "\\N"] for coord in coords]
    coords = np.asarray(coords)
    coords = np.asmatrix(coords)

    kms_per_radian = 6371.0086
    
    epsilon = 2 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_

    num_clusters = len(set(cluster_labels) - set([-1]))

    print( 'Clustered ' + str(len(coords)) + ' points to ' + str(num_clusters) + ' clusters')

    for n in range(num_clusters):
        one_cluster = coords[cluster_labels == n]
        print(one_cluster[:1])

def main():
    path = 'test.xlsx'
    dbscan(path)


if __name__ == '__main__':
    main()