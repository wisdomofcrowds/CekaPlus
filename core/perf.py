# -*- coding: utf-8 -*-

import core.data

class Evaluation:
    """
    Performance evaluation
    """
    def __init__(self, dataset = None):
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_accuracy(self):
        num_instance = self.dataset.get_instance_size()
        total = 0.0
        correct = 0.0
        for inst_id in range(1, num_instance + 1):
            inst = self.dataset.get_instance(inst_id)
            label_ids = inst.get_label_id_list()
            for label_id in label_ids:
                total += 1.0
                correct += inst.equal_integrated_true(label_id)
        return correct/total

    def get_accuracy_on_label(self, label_id):
        num_instance = self.dataset.get_instance_size()
        total = 0.0
        correct = 0.0
        for inst_id in range(1, num_instance + 1):
            inst = self.dataset.get_instance(inst_id)
            total += 1.0
            correct += inst.equal_integrated_true(label_id)
        return correct/total
