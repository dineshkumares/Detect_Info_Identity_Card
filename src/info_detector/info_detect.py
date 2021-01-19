import cv2
from info_detector.word_detect import WordDetect
from utility.info_utils import *
from utility.shape_utils import *
import glob
import copy


class InfoDetect(object):
    def __init__(self):
        return

    def detect_info(self, warped_img):
        detect_word = WordDetect()
        word_images, boxes = detect_word.detect_word(warped_img)

        boxes_temp = []
        for box in boxes:
            boxes_temp.append([box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]])
        boxes = boxes_temp
        self.eliminate_not_box(boxes)
        self.norm_boxes(boxes)

        ratios = np.asarray([
            [2.111, 4.285, 1.131, 3.00],
            [3.519, 3.157, 1.027, 1.875],
            [3.519, 1.935, 1.027, 1.622],
            [3.519, 1.667, 1.027, 1.263],
            [47.50, 1.290, 1.027, 1.043]
        ])

        infos = []  # consisting all infos field in Identity Card
        bounding_boxes = []

        classified = []
        h, w, _ = warped_img.shape
        classify_box(boxes, ratios, h, w, classified)
        delta = 3
        for classi, i in zip(classified, range(len(classified))):
            classi = connect_bounds(classi)
            self.eliminate_small_area_box(classi)
            remove_box_inside_other_box(classi)
            self.join_info(classi)
            for j in range(len(classi)):
                box = list(classi[j])
                # print(i, box)
                box[1] -= 3
                box[3] += 2 * delta
                infos.append(warped_img[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])])
                bounding_boxes.append(box)

        for i in range(0, len(bounding_boxes) - 1):
            for j in range(i + 1, len(bounding_boxes)):
                if bounding_boxes[i][1] > bounding_boxes[j][1]:
                    temp = bounding_boxes[i]
                    bounding_boxes[i] = bounding_boxes[j]
                    bounding_boxes[j] = temp
                    temp = infos[i]
                    infos[i] = infos[j]
                    infos[j] = temp

        return infos, bounding_boxes

    def eliminate_not_box(self, boxes):
        not_box = []

        for i in range(len(boxes)):
            if boxes[i][3] * 1.0 / boxes[i][2] > 2:
                not_box.append(i)

        for i in range(len(not_box)):
            del boxes[not_box[len(not_box) - 1 - i]]

    def norm_boxes(self, boxes):
        threshold_h = 70

        for i in range(len(boxes)):
            if boxes[i][3] > threshold_h:
                old_h = boxes[i][3]
                boxes[i] = [boxes[i][0], boxes[i][1], boxes[i][2], old_h / 2]  # box1
                box2 = [boxes[i][0], boxes[i][1] + old_h / 2, boxes[i][2], old_h / 2]
                boxes.insert(i, box2)
                i = i + 1

    def eliminate_small_area_box(self, boxes):
        threshold_area = 500
        i = 0
        while i < len(boxes):
            if get_area_of_box(boxes[i]) < threshold_area:
                del boxes[i]
            i += 1

    def join_info(self, boxes):
        alpha = 0.3
        i = 0
        while i < len(boxes):
            j = 0
            while j < len(boxes) and i < len(boxes):
                if i != j:

                    area_intersection = get_area_intersection(boxes[i], boxes[j])
                    area_min = min(get_area_of_box(boxes[i]), get_area_of_box(boxes[j]))

                    if area_intersection >= alpha * area_min:
                        boxes[i] = connect(boxes[i], boxes[j])
                        del boxes[j]
                    else:
                        j += 1
                    continue
                j += 1
            i += 1


if __name__ == "__main__":
    # for linkImg in glob.glob("../../data/test/output_1/*.jpg"):
    #     arr = linkImg.split("/")
    #     img_name = arr[len(arr) - 1]
    #     img = cv2.imread(linkImg)
    #     img_copy = copy.deepcopy(img)
    #     info_detector = InfoDetect()
    #     fields, boxes = info_detector.detect_info(img_copy)
    #     for wordImg, box in zip(fields, boxes):
    #         cv2.rectangle(img_copy, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255))
    #         cv2.imwrite("../../data/test/output_2/" + img_name, img_copy)

    linkImg = "../../data/train/output_1/3.jpg"
    img = cv2.imread(linkImg)
    img_copy = copy.deepcopy(img)
    info_detector = InfoDetect()
    fields, boxes = info_detector.detect_info(img_copy)
    for wordImg, box in zip(fields, boxes):
        cv2.rectangle(img_copy, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255))
        cv2.imwrite("result.png", img_copy)
