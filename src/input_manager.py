
class InputManager:

    def __init__(self):
        self.frame_x = 480
        self.frame_y = 640
        self.set_box([200,460,180,440], 0)


    def set_box(self, min_max_box, margin_ratio):
        self.min_max_box_orig = min_max_box
        min_x_orig = min_max_box[0]
        max_x_orig = min_max_box[1]
        min_y_orig = min_max_box[2]
        max_y_orig = min_max_box[3]
        self.width = max(max_x_orig-min_x_orig, max_y_orig-min_y_orig)
        self.margin = self.width * margin_ratio
        min_x = min_x_orig - self.margin
        max_x = min_x_orig + self.width + self.margin
        min_y = min_y_orig - self.margin
        max_y = min_y_orig + self.width + self.margin

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
        return (self.width + 2*self.margin, self.width + 2*self.margin)