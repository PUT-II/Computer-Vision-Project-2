from collections import Counter
from typing import List

import cv2 as cv
import numpy as np
from scipy import ndimage
from skimage.morphology import square, dilation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class SegmentationResult:
    def __init__(self, image, background_cluster):
        self.background_cluster = background_cluster
        self.image = image


class WordDescription:
    def __init__(self, from_: int, to: int, data: np.ndarray):
        self.from_ = from_
        self.to = to
        self.data = data


class RowDescription:
    def __init__(self, from_: int, to: int, data: np.ndarray, words: List[WordDescription]):
        self.from_ = from_
        self.to = to
        self.data = data
        self.words = words


def create_mask(mask_shape: tuple, output_shape: tuple, rows: List[RowDescription], moments: np.ndarray) -> np.ndarray:
    mask = np.zeros(mask_shape, dtype=np.uint8)
    for i, row in enumerate(rows):
        for word in row.words:
            start_point = (word.from_, row.from_)
            end_point = (word.to, row.to)
            mask = cv.line(mask, start_point, end_point, (i + 1,), 3)

    inv_moments = np.linalg.pinv(moments)
    warped_mask = cv.warpPerspective(mask, inv_moments, output_shape)
    return warped_mask


def get_clusters(images: List[np.ndarray]) -> List[np.ndarray]:
    result = []
    for i, image in enumerate(images):
        row_pixels = np.sum(image, axis=1)
        col_pixels = np.sum(image, axis=0)

        height, width = image.shape
        mask_rows = np.array([[r for _ in range(width)] for r in row_pixels])
        mask_cols = np.array([col_pixels for _ in range(height)])
        mask_comp = mask_rows * mask_cols * ndimage.convolve(image, [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ])

        c = KMeans(2).fit_predict(mask_comp.reshape(-1, 1))

        mask_clusters = c.reshape(mask_comp.shape)
        mask_clusters = dilation(mask_clusters, square(1))

        result.append(mask_clusters)
    return result


def get_segmentation_mask(images: List[np.ndarray], clusters: List[np.ndarray]) -> List[SegmentationResult]:
    result = []
    for (image_no_grid, mask_clusters) in zip(images, clusters):

        background_cluster = Counter(mask_clusters.flatten()).most_common(1)[0][0]
        marker_cluster = 1 - background_cluster

        rows, marked_rows = __find_groups(marker_cluster, mask_clusters)
        for g in marked_rows:
            a, b = g[0], g[-1]
            clipped = mask_clusters[a:b]
            cols, marked_cols = __find_groups(marker_cluster, clipped.T)

            if not len(marked_cols):
                rows[a:b] = background_cluster
                continue

            try:
                breaks = [(col_beg[0] - col_end[-1]) for col_end, col_beg in zip(marked_cols, marked_cols[1:])]
                max_connection_cols = np.max(breaks)
                min_connection_cols = np.min(breaks)

                for col_a, col_b in zip(marked_cols, marked_cols[1:]):
                    segments_distance = col_b[0] - col_a[-1]

                    distance_similarity_max = __calculate_similarity(max_connection_cols, segments_distance)
                    distance_similarity_min = __calculate_similarity(min_connection_cols, segments_distance)

                    if distance_similarity_max < 0.6 or distance_similarity_min > 0.2:
                        group_value = cols[col_a[-1] - 1][0]
                        cols[col_a[-1]:col_b[-1] + 1] = group_value

                rows[a:b] = cols.T
            except Exception:
                rows[a:b] = background_cluster

        result.append(SegmentationResult(rows, background_cluster))
    return result


def get_row_descriptions(images: List[np.ndarray], segmentation_results: List[SegmentationResult]):
    result: List[List[RowDescription]] = []
    for image, segmentation_result in zip(images, segmentation_results):
        segmented_image = segmentation_result.image
        background_cluster = segmentation_result.background_cluster
        rows = __extract_all_segments(image, segmented_image, background_cluster)
        result.append(rows)

    return result


def __extract_all_segments(image: np.ndarray, rows_mask, background_cluster) -> List[RowDescription]:
    all_rows: List[RowDescription] = []
    rows = __find_segments(rows_mask, background_cluster)
    for r in rows:
        words_in_row = __find_segments(r.data.T, background_cluster)
        words: List[WordDescription] = []

        for c in words_in_row:
            fragment = image[r.from_:r.to, c.from_:c.to]
            words.append(WordDescription(c.from_, c.to, fragment))

        fragment = image[r.from_:r.to]
        all_rows.append(RowDescription(r.from_, r.to, fragment, words))

    return all_rows


def __find_segments(matrix, background_cluster) -> List[WordDescription]:
    result: List[WordDescription] = []
    group: List[int] = []
    for num, dim in enumerate(matrix):
        if np.any(dim != background_cluster):
            group.append(num)
        elif len(group):
            result.append(WordDescription(group[0], group[-1], matrix[group]))
            group = []

    return result


def __find_groups(marker_cluster, matrix):
    group, marked = [], []
    for num, dim in enumerate(matrix):
        if np.any(dim == marker_cluster):
            group.append(num)
        elif len(group):
            marked.append(group)
            group = []

    lines_mask = np.zeros_like(matrix)
    for i, g in enumerate(marked, 1):
        a, b = g[0], g[-1]
        lines_mask[a:b] = i

    return lines_mask, marked


def __calculate_similarity(x, y):
    if x < y:
        x, y = y, x

    sim = cosine_similarity([[x, y]], [[y, x]])
    return sim[0][0]
