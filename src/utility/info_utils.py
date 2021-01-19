def is_connected(bound1, bound2):
    if bound1[3] < bound2[3]:  # box1's height > box2's height
        tmp = bound1
        bound1 = bound2
        bound2 = tmp

    x1, y1, w1, h1 = bound1
    x2, y2, w2, h2 = bound2

    up_y2 = y2
    down_y2 = y2 + h2

    threshold_ratio = 0.8

    # laws allow to connect 2 box
    # box 2 is below box 1
    if y1 <= up_y2 < y1 + threshold_ratio * h1:
        return True

    # box 2 is above box 1
    if y1 + h1 >= down_y2 > y1 + (1 - threshold_ratio) * h1:
        return True
    return False


# connect 2 bound return new bound
def connect(bound1, bound2):
    x1, y1, w1, h1 = bound1
    x2, y2, w2, h2 = bound2

    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)

    # new x, new y, new w, new h
    return x_min, y_min, x_max - x_min, y_max - y_min


# function connect 2 bound, which is near together
# return connected bounds
def connect_bounds(bound_rects):
    i = 0
    while i < len(bound_rects) - 1:
        if is_connected(bound_rects[i], bound_rects[i + 1]):
            bound_rects[i] = connect(bound_rects[i], bound_rects[i + 1])
            del bound_rects[i + 1]
        else:
            i += 1

    return bound_rects
