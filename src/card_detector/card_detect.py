import copy
from utility.card_utils import *
import glob


class CardDetect(object):
    _LOW_RANGE = [60, 21, 20]
    _HIGH_RANGE = [100, 255, 255]
    _MASK_DILATING = [10, 10]
    _MODE_FIND_CONTOUR = cv2.RETR_LIST
    _METHOD_FIND_CONTOUR = cv2.CHAIN_APPROX_TC89_L1

    def __init__(self):
        return

    def pre_processing(self, img):
        dst = copy.deepcopy(img)

        img_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, np.asarray(self._LOW_RANGE), np.asarray(self._HIGH_RANGE))
        # cv2.imwrite("mask.png", mask)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self._MASK_DILATING[0], self._MASK_DILATING[1]))
        mask = cv2.dilate(mask, element)
        # cv2.imwrite("mask_dilate.png", mask)

        return mask

    def detect_card(self, img):
        img1 = copy.deepcopy(img)
        img_processed = self.pre_processing(img1)

        contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # img_copy = copy.deepcopy(img)
        # img_contours = cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 2)
        # cv2.imwrite("contours.png", img_contours)

        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.02 * peri, True)   # rdp
        rect = cv2.minAreaRect(approx)  # return Box2D
        box = cv2.boxPoints(rect)       # 4 corners of the rectangle
        pts = np.int0(box)

        # box_pts = np.asarray(pts)
        # img_rdp = cv2.drawContours(img1, [box_pts], -1, (0, 0, 255), 8)
        # cv2.imwrite("min_area.png", img_rdp)

        # lay diem trong tam quoc huy => truyen vao de tim 4 dinh
        center = self.standardize_direction(img1)
        warped_img = four_point_transform(img1, pts, center)

        # cv2.drawContours(img1, [pts], -1, (0, 0, 255), 8)
        # cv2.imwrite("minAreaRect.png", img1)
        # cv2.imwrite("output.png", warped_img)

        return warped_img

    def standardize_direction(self, img):
        warped = copy.deepcopy(img)
        warped_inv = ~warped
        warped_hsv = cv2.cvtColor(warped_inv, cv2.COLOR_BGR2HSV)

        low_range = np.asarray([90 - 10, 120, 120])
        high_range = np.asarray([90 + 10, 255, 255])
        mask_dilating = [10, 10]
        mask = cv2.inRange(warped_hsv, low_range, high_range)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_dilating[0], mask_dilating[1]))
        mask = cv2.dilate(mask, element)
        # cv2.imwrite("warped_mask_dilate.png", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        # img_contours = cv2.drawContours(warped, contours, -1, (0, 0, 255), 2)
        # cv2.imwrite("warped_contours.png", img_contours)

        # tim diem trung tam
        moment = cv2.moments(contours[0])
        center = [round(moment['m10'] / moment['m00']), round(moment['m01'] / moment['m00'])]
        # cv2.circle(warped, (center[0], center[1]), 5, (0, 255, 0), -1)
        # cv2.imwrite("centroid.png", warped)
        # cv2.imshow("outline contour & centroid", warped)
        # cv2.waitKey(0)

        return center


if __name__ == "__main__":
    # for linkImg in glob.glob("../../data/test/*.jpg"):
    #     arr = linkImg.split("/")
    #     img_name = arr[len(arr) - 1]
    #     print(img_name)
    #     img = cv2.imread(linkImg)
    #     detector = CardDetect()
    #     result = detector.detect_card(img)
    #     cv2.imwrite("../../data/test/output_1/" + img_name, result)

    linkImg = "../../data/train/1.jpg"
    img = cv2.imread(linkImg)
    detector = CardDetect()
    result = detector.detect_card(img)
    cv2.imwrite("result.png", result)
