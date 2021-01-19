import cv2
import numpy as np
import copy


class WordDetect:
    _MIN_H = 10
    _MAX_H = 90

    def __init__(self):
        return

    def threshold(self, img):
        gray = np.min(img[:, :, :2], axis=2)    # BGR
        # cv2.imwrite("gray.png", gray)

        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # cv2.imwrite("gray_2.png", gray)

        gray_eq = cv2.equalizeHist(gray)
        # cv2.imwrite("gray_eq.png", gray_eq)

        binary = cv2.threshold(gray_eq, 60, 255, cv2.THRESH_BINARY_INV)[1]
        # cv2.imwrite("binary.png", binary)
        return binary

    def get_boxes(self, binary):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilate = cv2.dilate(binary, element, iterations=1)
        # dilate = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, element)
        # cv2.imwrite("dilate.png", dilate)

        # Drop small
        temp = copy.deepcopy(dilate)
        contours = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        text_binary = np.zeros(binary.shape, np.uint8)
        for contour in contours:
            x1, y1 = np.min(contour, axis=(0, 1))
            x2, y2 = np.max(contour, axis=(0, 1))
            if y2 - y1 < self._MIN_H or y2 - y1 > self._MAX_H:
                continue
            cv2.drawContours(text_binary, [contour], -1, 255)

        # cv2.imwrite("text_binary.png", text_binary)

        # Big connected components
        element[0][1] = 0
        element[2][1] = 0
        dilate = cv2.dilate(text_binary, element, iterations=4)
        # cv2.imwrite("dilate_2.png", dilate)

        # Find connected components
        temp = copy.deepcopy(dilate)
        contours = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        boxes = np.zeros(binary.shape, np.uint8)
        for contour in contours:
            x1, y1 = np.min(contour, axis=(0, 1))
            x2, y2 = np.max(contour, axis=(0, 1))
            if y2 - y1 < self._MIN_H:
                continue
            cv2.rectangle(boxes, (x1, y1), (x2, y2), 255)

        # cv2.imwrite("boxes.png", boxes)

        # Merge word
        contours = cv2.findContours(boxes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        bboxes = []
        # boxes_2 = np.zeros(binary.shape, np.uint8)
        for contour in contours:
            x1, y1 = np.min(contour, axis=(0, 1))
            x2, y2 = np.max(contour, axis=(0, 1))
            bboxes.append([[x1, y1], [x2, y2]])
            # cv2.rectangle(boxes_2, (x1, y1), (x2, y2), 255)

        # cv2.imwrite("boxes_2.png", boxes_2)

        return bboxes

    def detect_word(self, img):
        ROI = copy.deepcopy(img)
        binary = self.threshold(ROI)
        boxes = self.get_boxes(binary)

        word_images = []
        top = 1
        left = 8
        for box in boxes:
            # print box
            box[0][0] -= left
            box[0][1] -= top
            box[1][0] += left
            box[1][1] += top
            word_images.append(img[box[0][1]: box[1][1], box[0][0]: box[1][0], :])

            # print box
            cv2.rectangle(ROI, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 0, 255), thickness=3)

        # cv2.imwrite("word_boundingbox.png", ROI)
        return word_images, boxes


if __name__ == "__main__":
    linkImg = "../../data/train/output_1/3.jpg"
    img = cv2.imread(linkImg)
    detect_word = WordDetect()
    detect_word.detect_word(img)
