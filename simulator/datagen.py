# -*- coding: utf-8 -*-

import simulator.simdata as sd

class datagen:
    """
    A class to generate data
    """
    def __init__(self, multi_class):
        self.multi_class = multi_class
        self.inst_num = 0
        self.worker_num = 0
        self.label_num = 0
        self.dataset = sd.SimDataset()

    def create(self):
        # first create instances
        self.create_instances()

    def create_instances(self):
        for i in range(0, self.inst_num):
            inst  = sd.SimInstance(i+1)
            self.dataset.add_instance(inst)

generator = datagen(True)
generator.inst_num = 1000 # number of instances
generator.worker_number = 10 # number of workers
generator.label_num = 10 # number of label

generator.create_instances()


