from typing import List

import numpy as np
from matplotlib import pyplot as plt


def segment_to_digits(segment: np.array, verbose: int = 0) -> List[np.array]:
    from sklearn.cluster import OPTICS
    from statistics import mean

    nonzero_y, nonzero_x = np.where(segment > 0)
    nonzero_points = np.array(list(zip(nonzero_x, nonzero_y)))

    clt = OPTICS(min_samples=15,
                 max_eps=6,
                 xi=0.8,
                 min_cluster_size=50)

    try:
        clt.fit(nonzero_points)
    except Exception:
        return []

    labels = np.unique(clt.labels_)

    if labels.shape[0] < 6:
        clt.max_eps = 4
        clt.fit(nonzero_points)
        labels = np.unique(clt.labels_)
    elif labels.shape[0] > 6:
        clt.max_eps = 8
        clt.fit(nonzero_points)
        labels = np.unique(clt.labels_)

    outliers = []
    digits_coord = []
    digits_points = []
    for label in labels:
        points = nonzero_points[np.where(clt.labels_ == label)]

        if label == -1:
            outliers = points
            continue

        if verbose >= 1:
            digits_points.append(points)

        digits_coord.append((min(points[:, 1]), max(points[:, 1] + 1), min(points[:, 0]), max(points[:, 0]) + 1))

    if verbose >= 1:
        if len(outliers) > 2:
            plt.scatter(outliers[:, 0], outliers[:, 1], c='black')

        for points in digits_points:
            plt.scatter(points[:, 0], points[:, 1])
        plt.gca().invert_yaxis()
        plt.show()

        if verbose >= 2:
            print("Groups count:", len(digits_points))
            print("Min points in group:", min(g.shape[0] for g in digits_points))
            print("Mean points in group:", mean(g.shape[0] for g in digits_points))

    digits_coord.sort(key=lambda x: x[2])

    digits = []
    for d in digits_coord:
        digits.append(segment[d[0]:d[1], d[2]:d[3]])

    return digits


def digits_to_mnist_format(digits: List[np.array]) -> None:
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
            scaled_dim = max(1, int(max_size * d_proportion))
            d = resize(d, (max_size, scaled_dim), interpolation=INTER_AREA)
        else:
            d_proportion = d_width / d_height
            scaled_dim = max(1, int(max_size * d_proportion))
            d = resize(d, (scaled_dim, max_size), interpolation=INTER_AREA)

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


def load_clf_and_dataset(clf_pickle_path: str):
    import os
    import pickle
    from sklearn.datasets import fetch_openml
    from sklearn.svm import SVC

    if not os.path.isfile(clf_pickle_path):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

        clf = SVC().fit(X, y)

        with open(clf_pickle_path, 'wb') as f:
            pickle.dump(clf, f)
    else:
        with open(clf_pickle_path, 'rb') as f:
            clf = pickle.load(f)

    return clf


def predict_digits(clf, digits: List[np.array]) -> np.array:
    if not digits:
        return []
    reshaped = np.array([d.reshape(784) for d in digits])
    return clf.predict(reshaped)
