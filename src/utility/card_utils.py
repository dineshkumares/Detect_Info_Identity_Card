import cv2
import numpy as np


def four_point_transform(img, pts, center):
    rect = order_points(pts, center)
    (tl, tr, br, bl) = rect

    # compute width of wrapped perspective image
    width_top = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
    width_bottom = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
    max_width = max(int(width_top), int(width_bottom))

    # compute height
    height_left = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    height_right = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    max_height = max(int(height_right), int(height_left))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    mat = cv2.getPerspectiveTransform(rect, dst)
    warped_img = cv2.warpPerspective(img, mat, (max_width, max_height))
    warped_img = cv2.resize(warped_img, (950, int(max_height * 950 / max_width)))

    return warped_img


# initialize array with order : top-left, top-right, bottom-right, bottom-left
def order_points(pts, center):
    rect = np.zeros((4, 2), dtype="float32")

    # order points
    s = pts.sum(axis=1)  # y + x = c
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)  # y - x = c
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # calculate distance
    d_1 = np.sqrt((rect[0][0] - center[0]) ** 2 + (rect[0][1] - center[1]) ** 2)
    d_2 = np.sqrt((rect[1][0] - center[0]) ** 2 + (rect[1][1] - center[1]) ** 2)
    d_3 = np.sqrt((rect[2][0] - center[0]) ** 2 + (rect[2][1] - center[1]) ** 2)
    d_4 = np.sqrt((rect[3][0] - center[0]) ** 2 + (rect[3][1] - center[1]) ** 2)

    # top-left will has d=d_min
    d_min = min(d_1, d_2, d_3, d_4)
    if d_1 == d_min:
        rect_2 = rect
    elif d_2 == d_min:
        rect_2 = np.roll(rect, -1, axis=0)
    elif d_3 == d_min:
        rect_2 = np.roll(rect, -2, axis=0)
    else:
        rect_2 = np.roll(rect, -3, axis=0)

    return rect_2
