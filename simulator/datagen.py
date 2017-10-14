# -*- coding: utf-8 -*-

import numpy
import simulator.simdata as sd
import core.cio

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

    def create_instances(self):
        for i in range(1, self.inst_num + 1):
            inst  = sd.SimInstance(i)
            self.dataset.add_instance(inst)

    def create_workers(self):
        for i in range(1, self.worker_num + 1):
            worker = sd.SimWorker(i)
            self.dataset.add_worker(worker)

    def create_ml_uniK_uniDistrb_uniError(self, num_val, correct_rates):
        """
        create multi-label data set with only one number of values on each label
        , values are uniformed distributed and errors are uniformed distributed
        :param num_val: number of values on each label
        :param correct_rates: correct rates for each workers
        :return:
        """
        # first create instances
        self.create_instances()
        self.create_workers()
        self.dataset.set_worker_correct_rates(correct_rates)
        true_label_policies = dict()
        for label_id in range(1, self.label_num + 1):
            policy = sd.TrueLabelPolicy(label_id, num_val, self.inst_num)
            distrb = policy.uniform_distribution()
            true_label_policies.setdefault(label_id, distrb)
        self.dataset.set_true_label_distrib_policies(true_label_policies)
        instances = list()
        for id in range(1, self.inst_num + 1):
            instances.append(self.dataset.get_instance(id))
        # each worker label all instance
        for w in range(1, self.worker_num + 1):
            policy = sd.LabelingPolicy()
            policy.set_labeling_instances(instances)
            policy.set_label_info(self.dataset.label_info_dict)
            self.dataset.get_worker(w).labeling(policy)

def scenario1():
    generator = datagen(True)
    generator.inst_num = 100 # number of instances
    generator.worker_num = 10 # number of workers
    generator.label_num = 4# number of label
    num_val = 3# each label has K values

    # correct rate of workers are uniformed distributed in [0.6, 0.8]
    correct_rates = numpy.random.uniform(0.5, 0.6, generator.worker_num)
    generator.create_ml_uniK_uniDistrb_uniError(num_val, correct_rates)

    # save data
    out_resp_path = 'D:/Github/datasets/synth.resp'
    out_gold_path = 'D:/Github/datasets/synth.gold'
    core.cio.save_file(generator.dataset, out_resp_path, out_gold_path)

scenario1()


