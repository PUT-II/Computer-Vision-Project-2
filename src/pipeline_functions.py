import cv2 as cv
import numpy as np


def create_mask(mask_shape: tuple, output_shape: tuple, rows: list, moments) -> np.ndarray:
    mask = np.zeros(mask_shape, dtype=np.uint8)
    for i, row in enumerate(rows):
        for col in row['cols']:
            start_point = (col['from'], row['from'])
            end_point = (col['to'], row['to'])
            mask = cv.line(mask, start_point, end_point, (i + 1,), 3)

    inv_moments = np.linalg.pinv(moments)
    warped_mask = cv.warpPerspective(mask, inv_moments, output_shape)
    return warped_mask


def remove_grid_lines(img: np.ndarray) -> np.ndarray:
    sobel = None
    for i in range(20, 40, 5):
        image = cv.fastNlMeansDenoising(img, None, i, 5, 21)

        sobelx = cv.Sobel(image, cv.CV_8U, 1, 0, ksize=3)
        sobely = cv.Sobel(image, cv.CV_8U, 0, 1, ksize=3)

        sobelx = cv.fastNlMeansDenoising(sobelx, None, 15, 5, 21)
        sobely = cv.fastNlMeansDenoising(sobely, None, 15, 5, 21)

        sobel = cv.add(sobelx, sobely)
        _, sobel = cv.threshold(sobel, 32, 255, cv.THRESH_BINARY)

        if (np.count_nonzero(sobel) / (sobel.shape[0] * sobel.shape[1])) < 0.02:
            break

    kernel = np.ones((1, 8), np.uint8)
    erosion_1 = cv.erode(sobel, kernel, iterations=10)
    kernel = np.ones((5, 80), np.uint8)
    erosion_1 = cv.dilate(erosion_1, kernel, iterations=10)

    kernel = np.ones((8, 1), np.uint8)
    erosion_2 = cv.erode(sobel, kernel, iterations=20)
    kernel = np.ones((80, 5), np.uint8)
    erosion_2 = cv.dilate(erosion_2, kernel, iterations=10)

    mask = np.add(erosion_1, erosion_2)
    mask = cv.bitwise_not(mask)
    sobel_no_border = cv.bitwise_and(sobel, mask)

    return sobel_no_border


def straighten_page(img_: np.ndarray) -> np.ndarray:
    # Denoise image to improve contour detection
    # https://docs.opencv.org/master/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93
    image = cv.fastNlMeansDenoising(img_, None, 20, 7, 21)

    thresh = cv.adaptiveThreshold(src=image,
                                  maxValue=255,
                                  adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  thresholdType=cv.THRESH_BINARY_INV,
                                  blockSize=11,
                                  C=3)
    thresh = cv.fastNlMeansDenoising(thresh, None, 30, 7, 21)

    cv.imwrite("../test.png", thresh)
    contours_, _ = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    contours_flat = np.vstack(contours_).squeeze()

    rect = cv.minAreaRect(contours_flat)
    box = cv.boxPoints(rect)

    box_ordered = __order_points(box)

    img_warped = __four_point_transform(img_, box_ordered)
    return img_warped


# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

def __order_points(pts_):
    rect_ = np.zeros((4, 2), dtype="float32")
    s = pts_.sum(axis=1)
    rect_[0] = pts_[np.argmin(s)]
    rect_[2] = pts_[np.argmax(s)]
    diff = np.diff(pts_, axis=1)
    rect_[1] = pts_[np.argmin(diff)]
    rect_[3] = pts_[np.argmax(diff)]
    return rect_


def __four_point_transform(image_, rect_):
    (tl, tr, br, bl) = tuple(rect_)

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    moments = cv.getPerspectiveTransform(rect_, dst)
    warped = cv.warpPerspective(image_, moments, (max_width, max_height))
    return warped, moments
