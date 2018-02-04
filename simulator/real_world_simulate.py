# -*- coding: utf-8 -*-

import numpy
import simulator.simdata as sd
import core.cio
import core.data
import core.utils
import sklearn.decomposition

class real_world_simulate:

    def __init__(self):
        self.dataset = None

    def create_data_set(self, gold_path):
        dataset = sd.SimDataset()
        gold_file = open(gold_path)
        for line in gold_file:
            strs = line.split()
            if len(strs) <= 1:
                gold_file.close()
                print('Error formatted gold file', end='\n')
                return None

            inst_id = dataset.fetch_instance_id_by_name(strs[0])
            inst = dataset.get_instance(inst_id)
            if inst == None:
                inst = sd.SimInstance(inst_id)
                dataset.add_instance(inst)

            label_id = 1
            label_val = 0
            if (len(strs) + 1) == core.data.Label.SINGLE_LABLE:
                label_id = dataset.fetch_label_id_by_name('SINGLE_LABEL')
                label_val = dataset.fetch_label_val_id_by_name(label_id, strs[1])  # label id is 1
            if (len(strs) + 1) == core.data.Label.MULTI_LABEL:
                label_id = dataset.fetch_label_id_by_name(strs[1])
                label_val = dataset.fetch_label_val_id_by_name(label_id, strs[2])
            true_label = inst.get_true_label(label_id)
            if true_label == None:
                true_label = sd.SimLabel(label_id)
                true_label.inst_id = inst.id
                true_label.worker_id = sd.SimWorker.GOLD
                inst.add_true_label(true_label)
            true_label.val = label_val

        gold_file.close()
        dataset.label_info_confirm()
        dataset.create_from_files = True
        self.dataset = dataset
        return self.dataset

class SimUniformIndependentWorker(core.data.Worker):

    def __init__(self, id = 0):
        core.data.Worker.__init__(self, id)
        self.correct_rate = 0.0

    def labeling(self, dataset):
        count = 0
        # for each label_id
        for (label_id, label_vals) in dataset.label_info_dict.items():
            # first we determine the correct instances
            inst_num = dataset.get_instance_size()
            correct_num = int(inst_num * self.correct_rate)
            correct = core.utils.rand_int_no_repeat(1, inst_num, correct_num)
            for id in range(1, inst_num + 1):
                inst = dataset.get_instance(id)
                true_label = inst.get_true_label(label_id)
                true_val = true_label.val
                noisy_label = sd.SimLabel(label_id)
                noisy_label.inst_id = inst.id
                noisy_label.worker_id = self.id
                if inst.id in correct:
                    noisy_label.val = true_val
                else:
                    noisy_label.val = core.utils.exclusive_rand(true_val, label_vals)[0]
                self.add_label(noisy_label)
                inst.add_noisy_label(noisy_label)
                count += 1
                print(str(count) + '. worker (' + str(self.id) + ') labeling instance (' \
                      + str(inst.id) + ') label (' + str(label_id) + ') as value (' + str(noisy_label.val) + ')',
                      end='\n')


class SimBinaryBiasedWroker(core.data.Worker):
    def __init__(self, id=0):
        core.data.Worker.__init__(self, id)
        self.positive_correct_rate = 0.0
        self.negative_correct_rate = 0.0

    def labeling(self, dataset):
        count = 0
        # for each label_id
        positive_num_list = list()
        negative_num_list = list()
        num_labels = dataset.get_label_id_size()
        for i in range(0, num_labels + 1):
            positive_num_list.append(0)
            negative_num_list.append(0)
        for (label_id, label_vals) in dataset.label_info_dict.items():
            positive_insts = list()
            negative_insts = list()
            inst_num = dataset.get_instance_size()
            for id in range(1, inst_num + 1):
                inst = dataset.get_instance(id)
                true_label = inst.get_true_label(label_id)
                true_val = true_label.val
                if true_val ==  2:
                    positive_num_list[label_id] += 1
                    positive_insts.append(id)
                else:
                    negative_num_list[label_id] += 1
                    negative_insts.append(id)
            print('Label ' + str(label_id) + ' positive (' + str(positive_num_list[label_id]) + ') negative (' \
                  + str( negative_num_list[label_id]) + ') ', end='\n')
            positive_correct_num = int(positive_num_list[label_id] * self.positive_correct_rate)
            negative_correct_num = int(negative_num_list[label_id] * self.negative_correct_rate)
            positive_correct_pos = core.utils.rand_int_no_repeat(0, positive_num_list[label_id] - 1, positive_correct_num)
            negative_correct_pos = core.utils.rand_int_no_repeat(0, negative_num_list[label_id] - 1, negative_correct_num)
            positive_correct_insts = list()
            negative_correct_insts = list()
            for i in range(0, positive_num_list[label_id]):
                if i in positive_correct_pos:
                    positive_correct_insts.append(positive_insts[i])
            for i in range(0, negative_num_list[label_id]):
                if i in negative_correct_pos:
                    negative_correct_insts.append(negative_insts[i])

            for id in range(1, inst_num + 1):
                inst = dataset.get_instance(id)
                true_label = inst.get_true_label(label_id)
                true_val = true_label.val
                noisy_label = sd.SimLabel(label_id)
                noisy_label.inst_id = inst.id
                noisy_label.worker_id = self.id
                if inst.id in positive_correct_insts:
                    noisy_label.val = true_val
                elif inst.id in negative_correct_insts:
                    noisy_label.val = true_val
                else:
                    if true_val == 1:
                        noisy_label.val = 2
                    else:
                        noisy_label.val = 1
                self.add_label(noisy_label)
                inst.add_noisy_label(noisy_label)
                count += 1
#                print(str(count) + '. worker (' + str(self.id) + ') labeling instance (' \
#                      + str(inst.id) + ') label (' + str(label_id) + ') as value (' + str(noisy_label.val) + ')',
#                      end='\n')

class SimUniformCorrelationWorker(core.data.Worker):
    def __init__(self, id=0):
        core.data.Worker.__init__(self, id)
        self.consistency_probs = list()
        self.correlated_correct_rate = 0.0
        self.non_correlated_correct_rate = 0.0

    def set_consistency_info(self, initial=0.9, step=0.1, maxnum=3):
        self.consistency_probs.append(initial)
        for i in range (1, maxnum):
            self.consistency_probs.append(self.consistency_probs[i-1]-step)

    def labeling(self, dataset):
        num_labels = dataset.get_label_id_size()
        num_inst = dataset.get_instance_size()
        correlation = numpy.ndarray(shape=(num_inst, num_labels ), dtype=float, order='C')
        correlation.fill(0)
        for id in range(1, num_inst + 1):
            inst = dataset.get_instance(id)
            for label_id in range (1, num_labels + 1):
                true_label = inst.get_true_label(label_id)
                true_val = true_label.val
                correlation[id - 1][label_id  - 1] = true_val
        cov = numpy.cov(correlation.T)
        #print(cov)
        # pca = sklearn.decomposition.PCA(n_components='mle')
        # pcaresult = pca.fit(correlation)
        # print(pca.get_covariance())
        cov_dict = dict()
        for i in range (0, num_labels):
            for j in range(i + 1, num_labels):
                t = list()
                t.append(i)
                t.append(j)
                cov_dict.setdefault(abs(cov[i, j]), t)
        sorted_keys = sorted(cov_dict, reverse=True)
        print(sorted_keys)
        # correlated_labeling
        num_consistency = len(self.consistency_probs)
        label_proceed = list()
        label_proceed.append(True)
        for i in range (1, num_labels + 1):
            label_proceed.append(False)
        for r in range (0, num_consistency):
            pos = cov_dict.get(sorted_keys[r])
            label_id = pos[0] + 1
            print('correlated label = (' + str(pos[0] + 1) + ',' + str(pos[1] + 1) +')')
            if label_proceed[label_id] == False:
                # labeling
                self._labeling(label_id, dataset, self.correlated_correct_rate)
                label_proceed[label_id] = True
            consistency_list = self.compute_consistency( label_id, dataset)
            error_num = int((1 - self.consistency_probs[r]) * len(consistency_list)-1)
            error_list = core.utils.rand_int_no_repeat(1, len(consistency_list), error_num)
            for k in range(1, len(consistency_list)):
                if k in error_list:
                    if consistency_list[k] == True:
                        consistency_list[k] = False
                    else:
                        consistency_list[k] = True
            label_id = pos[1] + 1
            if label_proceed[label_id] == False:
                # labeling
                self._labeling_with_consistency(label_id, dataset, consistency_list)
                label_proceed[label_id] = True
        # non-correlated labeling
        for label_id in range(1, num_labels + 1):
            if (label_proceed[label_id] == False):
                self._labeling(label_id, dataset, self.non_correlated_correct_rate)

    def _labeling(self, label_id, dataset, correct_rate):
        inst_num = dataset.get_instance_size()
        correct_num = int(inst_num * correct_rate)
        correct = core.utils.rand_int_no_repeat(1, inst_num, correct_num)
        label_vals = dataset.label_info_dict[label_id]
        for id in range(1, inst_num + 1):
            inst = dataset.get_instance(id)
            true_label = inst.get_true_label(label_id)
            true_val = true_label.val
            noisy_label = sd.SimLabel(label_id)
            noisy_label.inst_id = inst.id
            noisy_label.worker_id = self.id
            if inst.id in correct:
                noisy_label.val = true_val
            else:
                noisy_label.val = core.utils.exclusive_rand(true_val, label_vals)[0]
            self.add_label(noisy_label)
            inst.add_noisy_label(noisy_label)

    def _labeling_with_consistency(self, label_id, dataset, consistency_list):
        inst_num = dataset.get_instance_size()
        label_vals = dataset.label_info_dict[label_id]
        for id in range(1, inst_num + 1):
            inst = dataset.get_instance(id)
            true_label = inst.get_true_label(label_id)
            true_val = true_label.val
            noisy_label = sd.SimLabel(label_id)
            noisy_label.inst_id = inst.id
            noisy_label.worker_id = self.id
            if consistency_list[id] == True:
                noisy_label.val = true_val
            else:
                noisy_label.val = core.utils.exclusive_rand(true_val, label_vals)[0]
            self.add_label(noisy_label)
            inst.add_noisy_label(noisy_label)

    def compute_consistency(self, label_id, dataset):
        consistency_list = list()
        consistency_list.append(False)
        inst_num = dataset.get_instance_size()
        for id in range(1, inst_num + 1):
            inst = dataset.get_instance(id)
            true_label = inst.get_true_label(label_id)
            true_val = true_label.val
            noisy_label_val = self.get_label_val_for_inst(inst.id, label_id)
            if noisy_label_val == true_val:
                consistency_list.append(True)
            else:
                consistency_list.append(False)
        return consistency_list


gold_path_o = 'D:/Github/datasets/emotions/emotions-train.gold'
resp_path = 'D:/Github/datasets/emotions/emotions-train.resp'
gold_path = 'D:/Github/datasets/emotions/emotions-train.gold'
rws = real_world_simulate()
rws.create_data_set(gold_path_o)
#
#create Type1 worker
num_type1 = 9
correct_rates = numpy.random.uniform(0.6, 0.75, num_type1)
for workid in range(1, num_type1 + 1):
    worker = SimUniformIndependentWorker(workid)
    worker.correct_rate = correct_rates[workid - 1]
    rws.dataset.add_worker(worker)
    worker.labeling(rws.dataset)

#create Type2 worker
#num_type2 =4
#positive_correct_rate = 0.45
#negative_correct_rate = 0.85

#for workid in range(1, num_type2 + 1):
#    worker = SimBinaryBiasedWroker(workid)
#    worker.positive_correct_rate = positive_correct_rate
#    worker.negative_correct_rate = negative_correct_rate
#    rws.dataset.add_worker(worker)
#    worker.labeling(rws.dataset)

#create Type3 worker
#num_type3 = 9
#non_correlated_correct_rates = numpy.random.uniform(0.6, 0.7, num_type3)
#correlated_correct_rates = numpy.random.uniform(0.70, 0.80, num_type3)

#for workid in range(1, num_type3 + 1):
#    worker = SimUniformCorrelationWorker(workid)
#    worker.correlated_correct_rate = correlated_correct_rates[workid - 1]
#    worker.non_correlated_correct_rate = non_correlated_correct_rates[workid - 1]
#    worker.set_consistency_info(0.99, 0.05, 2)
#    rws.dataset.add_worker(worker)
#    worker.labeling(rws.dataset)

core.cio.save_file(rws.dataset, resp_path, gold_path)
