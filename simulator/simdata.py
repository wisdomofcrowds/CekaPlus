# -*- coding: utf-8 -*-

import core.data

class TrueLabelPolicy:
    """
    Policy on how to generate true label
    """
    def __init__(self, label_id, val_num = 2):
        self.label_id = label_id
        self.val_num = val_num
        self.prob = dict()
        for k in range(0, val_num):
            self.prob.setdefault(k + 1, 1.0/val_num)

    def set_val_prob(self, val, prob):
        self.prob[val] = prob

class SimLabel(core.data.Label):

    def __init__(self, id = 0):
        core.data.Label.__init__(self, id)

class SimWorker(core.data.Worker):
    def __init__(self, id = 0):
        core.data.Worker.__init__(self, id)

class SimInstance(core.data.Instance):
    def __init__(self, id = 0):
        core.data.Instance.__init__(self, id)

class SimDataset(core.data.Dataset):
    def __init__(self):
        core.data.Dataset.__init__(self)

