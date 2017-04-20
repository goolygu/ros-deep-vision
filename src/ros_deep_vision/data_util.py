
from distribution import *

def closest_value(mat, p_x, p_y):
    min_dist = 1000000;
    closest_value = 0
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            if mat[x,y] > 0:
                dist = (x-p_x)**2 + (y-p_y)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_value = mat[x,y]

    return closest_value

# This is for depth image, find closest value to point px py in mat that is not zero
def closest_value_fast(mat, p_x, p_y):
    # create a sprial see http://stackoverflow.com/questions/398299/looping-in-a-spiral
    X = max(mat.shape[0] - p_x, p_x)*2
    Y = max(mat.shape[1] - p_y, p_y)*2
    x = y = 0
    dx = 0
    dy = -1
    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
            # print (x, y),
            x_cord, y_cord = x+p_x, y+p_y
            if x_cord >= 0 and x_cord < mat.shape[0] and y_cord >=0 and y_cord < mat.shape[1] and mat[x_cord, y_cord] > 0:
                return mat[x_cord, y_cord]

        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy

def closest_pc_value_fast(pc, p_x, p_y, max_width):
    # create a sprial see http://stackoverflow.com/questions/398299/looping-in-a-spiral

    if np.isnan(p_x) or np.isnan(p_y):
        return [float('nan'), float('nan'), float('nan')]

    p_x = int(p_x)
    p_y = int(p_y)

    X = max(pc.shape[0] - p_x, p_x)*2
    Y = max(pc.shape[1] - p_y, p_y)*2
    x = y = 0
    dx = 0
    dy = -1
    x_cord = 0
    y_cord = 0

    for i in range(max(X, Y)**2):
        if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
            # print (x, y),
            if abs(x) > max_width or abs(y) > max_width:
                return [float('nan'), float('nan'), float('nan')]
            x_cord, y_cord = x+p_x, y+p_y

            point = pc[x_cord, y_cord]
            if x_cord >= 0 and x_cord < pc.shape[0] and y_cord >=0 and y_cord < pc.shape[1] and not (np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]) ):
                return pc[x_cord, y_cord].tolist()

        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy

def state_list_to_dist(state_list):
    dist = Distribution()
    for state in state_list:
        # print "state", state.name
        if state.type == "cnn":
            dist.set_tree_feature(state.name)
    return dist


class Dic2:

    def __init__(self):
        self.dic = {}

    def __repr__(self):
        return repr(self.dic)

    def add(self,k1,k2,value):
        if not k1 in self.dic:
            self.dic[k1] = {}

        self.dic[k1][k2] = value

    def has(self,k1,k2):
        if k1 in self.dic:
            if k2 in self.dic[k1]:
                return True

        return False

    def get(self,k1,k2):

        return self.dic[k1][k2]

    def set(self,k1,k2,value):
        if self.has(k1,k2):
            self.dic[k1][k2] = value
            return True
        else:
            return False

    def keys1(self):
        return self.dic.keys()

    def keys2(self, k1):
        return self.dic[k1].keys()

    def get_sublist(self, k1):
        sublist = []
        for k2 in self.dic[k1]:
            value = self.dic[k1][k2]
            sublist.append((k2,value))
        return sublist

    def get_list(self):
        tuple_list = []
        for k1 in self.dic:
            for k2 in self.dic[k1]:
                value = self.dic[k1][k2]
                tuple_list.append((k1,k2,value))
        return tuple_list

    def combine(self, dic_other):
        for k1 in dic_other.dic:
            for k2 in dic_other.dic[k1]:
                value = dic_other.dic[k1][k2]
                self.add(k1,k2,value)

    def clear(self):
        self.dic.clear()
