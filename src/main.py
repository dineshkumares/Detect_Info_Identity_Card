import glob
import cv2
import numpy as np
from card_detector.card_detect import CardDetect
from recognition.recognize import Recognize
from info_detector.info_detect import InfoDetect
from PIL import ImageFont, ImageDraw, Image
import copy

card_detect = CardDetect()
info_detect = InfoDetect()
recognize = Recognize()

if __name__ == "__main__":
    index = 1
    for i in range(1, 2):
        linkImg = "../data/" + str(i) + ".jpg"
        img = cv2.imread(linkImg)

        comp = linkImg.split('/')
        folder = '/'.join(comp[:len(comp) - 1]) + "/output/"
        name = comp[-1].split(".")[0]

        card = card_detect.detect_card(img)

        if "NoneType" not in str(type(card)):
            fields, boxes = info_detect.detect_info(copy.deepcopy(card))
            texts = []
            for wordImg, box in zip(fields, boxes):
                if wordImg.shape[0] < 35:
                    continue

                text = recognize.read_word(wordImg)
                if len(text) < 3:
                    continue

                cv2.imwrite("wordImage/" + str(index) + ".png", wordImg)
                index = index + 1

                # print("label: ", text)
                texts.append([text, (box[0], box[1])])
                cv2.rectangle(card, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255))
                # cv2.imwrite("result.png", card)

            # print("----------------------------------------------------")
            img = Image.fromarray(card)
            font = ImageFont.truetype("rcnn/fonts/arial.ttf", 24)
            draw = ImageDraw.Draw(img)
            for text, pos in texts:
                print(text)
                draw.text(pos, text, font=font, fill=(0, 0, 255, 0))
            img = np.array(img)
            # print("----------------------------------------------------")
