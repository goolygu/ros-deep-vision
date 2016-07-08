import math
class InputManager:

    def __init__(self):
        self.frame_x = 480
        self.frame_y = 640
        self.min_box = 300
        self.set_box([200,460,180,440], 0)


    def set_box(self, min_max_box, margin_ratio):
        self.min_max_box_orig = min_max_box
        min_x_orig = min_max_box[0]
        max_x_orig = min_max_box[1]
        min_y_orig = min_max_box[2]
        max_y_orig = min_max_box[3]
        self.width = max(max_x_orig-min_x_orig, max_y_orig-min_y_orig)
        self.margin = round(self.width * margin_ratio)
        if self.width + 2*self.margin > self.frame_x:
            self.margin = int((self.frame_x - self.width)/2.0)

        if self.margin < 0:
            center_y = (min_y_orig + max_y_orig)/2
            min_x = 0
            max_x = self.frame_x - 1
            min_y = center_y - self.frame_x/2
            max_y = center_y + self.frame_x/2
            self.min_max_box = [min_x, max_x, min_y, max_y]
            return
        min_x = min_x_orig - self.margin
        max_x = min_x_orig + self.width + self.margin
        min_y = min_y_orig - self.margin
        max_y = min_y_orig + self.width + self.margin

        if (max_x - min_x) < self.min_box:
            min_x -= (self.min_box - (max_x - min_x))/2
            max_x += (self.min_box - (max_x - min_x))/2
            min_y -= (self.min_box - (max_y - min_y))/2
            max_y += (self.min_box - (max_y - min_y))/2

        # consider corner conditions
        if max_x > self.frame_x:
            shift = max_x - self.frame_x
            min_x -= shift
            max_x = self.frame_x
        if min_x < 0:
            shift = -min_x
            max_x += shift
            min_x = 0

        if max_y > self.frame_y:
            shift = max_y - self.frame_y
            min_y -= shift
            max_y = self.frame_y
        if min_y < 0:
            shift = -min_y
            max_y += shift
            min_y = 0

        self.min_max_box = [min_x, max_x, min_y, max_y]

    def crop(self, frame):
        min_x = self.min_max_box[0]
        max_x = self.min_max_box[1]
        min_y = self.min_max_box[2]
        max_y = self.min_max_box[3]
        return frame[min_x:max_x,min_y:max_y,:]

    def get_crop_bias(self):
        min_x = self.min_max_box[0]
        min_y = self.min_max_box[2]
        return (min_x, min_y)

    def get_after_crop_size(self):
        return (self.min_max_box[1] - self.min_max_box[0], self.min_max_box[3] - self.min_max_box[2])


        # return (self.width + 2*self.margin, self.width + 2*self.margin)
