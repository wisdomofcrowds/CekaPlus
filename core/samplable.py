# -*- coding: utf-8 -*-

class Samplable:
    """
    The base class for the object that will use sampling method
    """
    def __init__(self ):
        self.current = -1
        self.objs = []

    def append(self, obj):
        self.objs.append(obj)
        self.current += 1

    def setV(self, val, index = 0):
        if (index == 0):
            index = self.current
        self.objs[index] = val

    def getV(self, index = 0):
        if (index == 0):
            index = self.current
        return self.objs[index]

    def print_obj(self, index = 0):
        if (index == 0):
            index = self.current
        print(self.objs[index], end=' ')


class RealV(Samplable):

    def __init__(self, val):
        Samplable.__init__(self)
        self.append(val)

    def getV(self, index = 0):
        if (index == 0):
            index = self.current
        return float(self.objs[index])


class IntV(Samplable):
    def __init__(self, val):
        Samplable.__init__(self)
        self.append(val)

    def getV(self, index=0):
        if (index == 0):
            index = self.current
        return int(self.objs[index])
