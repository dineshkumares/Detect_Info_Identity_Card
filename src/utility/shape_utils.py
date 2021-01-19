from shapely.geometry import Polygon
import numpy as np
import cv2


def remove_box_inside_other_box(boxes):
    is_removed = len(boxes) * [False]

    for i1 in range(len(boxes)):
        for i2 in range(len(boxes)):
            polygon1 = create_polygon_from_box(boxes[i1])
            polygon2 = create_polygon_from_box(boxes[i2])

            if i1 != i2:
                if polygon1.contains(polygon2):
                    is_removed[i2] = True
    i = 0
    while i < len(boxes):
        if is_removed[i]:
            del boxes[i]
            del is_removed[i]

        else:
            i += 1


def create_closed_polygon_from_list(list):
    tmp = [i for i in list]
    tmp.append(list[0])  # create closed polygon
    return Polygon(tmp)


def create_polygon_from_ratio(ratio, h_img, w_img):
    ratio_x, ratio_y, ratio_x1, ratio_y1 = ratio

    x = int(w_img / ratio_x)
    y = int(h_img / ratio_y)
    x1 = int(w_img / ratio_x1)
    y1 = int(h_img / ratio_y1)

    coords = [[x, y], [x1, y], [x1, y1], [x, y1]]
    return create_closed_polygon_from_list(coords)


def create_polygon_from_box(box):
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h

    coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return create_closed_polygon_from_list(coords)


# box ~ [x0, y0, w, h]
def classify_box(boxes, ratios, h_img, w_img, classified):
    classified.append([])  # id
    classified.append([])  # name
    classified.append([])  # birth
    classified.append([])  # country
    classified.append([])  # residence registration

    polygon_ratios = []
    # create polygon for ratios
    for ratio in ratios:
        polygon_ratios.append(create_polygon_from_ratio(ratio, h_img, w_img))

    # classify each box for fields
    for box in boxes:
        areas_intersections = []
        polygon_box = create_polygon_from_box(box)

        for i in range(len(polygon_ratios)):
            intersections = polygon_box.intersection(polygon_ratios[i])
            areas_intersections.append(intersections.area)

        index_max = np.argmax(areas_intersections)

        # if has one or more field intersect with box
        # item() => convert np.int64 to int
        if areas_intersections[index_max.item()] > 0 and areas_intersections[index_max.item()] > polygon_box.area / 2:
            classified[index_max].append(box)


def get_area_intersection(box1, box2):
    polygon1 = create_polygon_from_box(box1)
    polygon2 = create_polygon_from_box(box2)

    tmp = polygon1.intersection(polygon2)
    return tmp.area


def get_area_of_box(box):
    return create_polygon_from_box(box).area
