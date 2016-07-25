

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
