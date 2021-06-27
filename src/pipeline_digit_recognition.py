from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt


def segment_to_digits(segment: np.array) -> List[np.array]:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS
    from statistics import mean

    non_zero_y, non_zero_x = np.where(segment > 0)
    non_zero_points = np.array(list(zip(non_zero_x, non_zero_y)))

    digits_coord = []
    groups = []
    # clt = DBSCAN(min_samples=30, eps=5, n_jobs=-1)
    # clt = AgglomerativeClustering(n_clusters=None, distance_threshold=300)
    clt = OPTICS(min_samples=30, eps=10, min_cluster_size=100, cluster_method='dbscan')
    clt.fit(non_zero_points)

    for label in np.unique(clt.labels_):
        if label == -1:  # W grupie -1 są outliery
            continue

        points = non_zero_points[np.where(clt.labels_ == label)]
        plt.scatter(points[:, 0], points[:, 1])
        digits_coord.append((min(points[:, 1]), max(points[:, 1] + 1), min(points[:, 0]), max(points[:, 0]) + 1))

        groups.append(points)
    plt.gca().invert_yaxis()
    plt.show()

    print("Groups count:", len(groups))
    print("Min points in group:", min(g.shape[0] for g in groups))
    print("Mean points in group:", mean(g.shape[0] for g in groups))

    digits_coord.sort(key=lambda x: x[2])

    digits = []
    for d in digits_coord:
        digits.append(segment[d[0]:d[1], d[2]:d[3]])

    return digits


def digits_to_mnist_format(digits: List[np.array]):
    from cv2 import resize, INTER_AREA, copyMakeBorder, BORDER_CONSTANT, dilate
    import math

    for i in range(len(digits)):
        d = digits[i]

        d_height, d_width = d.shape

        sample_size = 28
        border_size = 4
        max_size = sample_size - (border_size * 2)

        if d_width > d_height:
            d_proportion = d_height / d_width
            d = resize(d, (max_size, int(max_size * d_proportion)), interpolation=INTER_AREA)
        else:

            d_proportion = d_width / d_height
            d = resize(d, (int(max_size * d_proportion), max_size), interpolation=INTER_AREA)

        border_v = (sample_size - d.shape[0]) / 2
        border_v_T = math.ceil(border_v)
        border_v_B = math.floor(border_v)

        border_h = (sample_size - d.shape[1]) / 2
        border_h_L = math.ceil(border_h)
        border_h_R = math.floor(border_h)

        d = copyMakeBorder(d, border_v_T, border_v_B, border_h_L, border_h_R, BORDER_CONSTANT)

        kernel = np.ones((2, 2), np.uint8)
        d = dilate(d, kernel, iterations=1)

        digits[i] = d


def load_clf_and_dataset(clf_pickle_path: str, dataset_pickle_path: str) -> (object, Tuple):
    import os
    import pickle
    from sklearn.datasets import fetch_openml
    from sklearn.svm import SVC

    if not os.path.isfile(dataset_pickle_path):
        X, y = data = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        with open(dataset_pickle_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(dataset_pickle_path, 'rb') as f:
            X, y = pickle.load(f)

    if not os.path.isfile(clf_pickle_path):
        clf = SVC()
        clf.fit(X, y)
        with open(clf_pickle_path, 'wb') as f:
            pickle.dump(clf, f)
    else:
        with open(clf_pickle_path, 'rb') as f:
            clf = pickle.load(f)
    return clf, (X, y)


def predict_digits(clf, digits: List[np.array]) -> np.array:
    reshaped = np.array([d.reshape(784) for d in digits])
    return clf.predict(reshaped)
