# -*- coding: utf-8 -*-
import numpy
import math
import core.data
import core.utils

class TrueLabelPolicy:
    """
    Policy on how to generate true label
    """
    def __init__(self, label_id, val_num, inst_num):
        self.label_id = label_id
        self.val_num = val_num
        self.inst_num = inst_num

    def uniform_distribution(self):
        """
        label values are uniform distributed across the instances
        :return: distribution map of instances w.r.t label values
        """
        distrib = dict()
        prob = 1.0/self.val_num
        inst_ids = [x for x in range(1, self.inst_num + 1)]
        inst_ids = numpy.random.permutation(inst_ids)
        group_num = self.inst_num // self.val_num
        last_group_num = self.inst_num - (self.val_num - 1) * group_num
        for i in range (0, self.val_num - 1):
            distrib.setdefault(i + 1, list())
            for j in range(0, group_num):
                distrib[i + 1].append(inst_ids[i * group_num + j])
        distrib.setdefault(self.val_num, list())
        for j in range(0, last_group_num):
            distrib[self.val_num].append(inst_ids[(self.val_num - 1) * group_num + j])
        return distrib

class LabelingPolicy:
        """
        policy on how to label data set
        """
        ERROR_UNIFORM = 0

        def __init__(self):
            self.labeling_instances = None
            self.errordistrib = self.ERROR_UNIFORM
            self.label_info = None
            pass

        def set_labeling_instances(self, instances):
            self.labeling_instances = instances

        def set_label_info(self, info):
            self.label_info = info

class SimLabel(core.data.Label):

    def __init__(self, id = 0):
        core.data.Label.__init__(self, id)


class SimWorker(core.data.Worker):
    def __init__(self, id = 0):
        core.data.Worker.__init__(self, id)
        self.correct_rate = 0.0

    def labeling(self, policy):
        count = 0
        if policy.errordistrib == LabelingPolicy.ERROR_UNIFORM:
            # for each label_id
            label_info = policy.label_info
            for (label_id, label_vals) in label_info.items():
                # first we determine the correct instances
                correct_num = int(len(policy.labeling_instances) * self.correct_rate)
                correct = core.utils.rand_int_no_repeat(1, len(policy.labeling_instances), correct_num)
                for inst in policy.labeling_instances:
                    true_label = inst.get_true_label(label_id)
                    true_val = true_label.val
                    noisy_label = SimLabel(label_id)
                    noisy_label.inst_id = inst.id
                    noisy_label.worker_id = self.id
                    if inst.id in correct:
                        noisy_label.val = true_val
                    else:
                        noisy_label.val = core.utils.exclusive_rand(true_val, label_vals)[0]
                    self.add_label(noisy_label)
                    inst.add_noisy_label(noisy_label)
                    count += 1
                    print(str(count) + '. worker ('+str(self.id) +') labeling instance ('\
                          +str(inst.id)+') label ('+ str(label_id)+') as value (' + str(noisy_label.val) +')', end='\n')


class SimInstance(core.data.Instance):
    def __init__(self, id = 0):
        core.data.Instance.__init__(self, id)


class SimDataset(core.data.Dataset):
    def __init__(self):
        core.data.Dataset.__init__(self)

    def set_true_labels_ml(self, num_labels, distributions):
        pass

    def set_worker_correct_rates(self, correct_rates):
        num_worker = len(self.worker_dict)
        for id in range(1, num_worker + 1):
            self.worker_dict[id].correct_rate = correct_rates[id - 1]

    def set_true_label_distrib_policies(self, policies):
        count = 0
        for (label_id, distrib) in policies.items():
            for (val, lst) in distrib.items():
                for e in lst:
                    label = SimLabel(label_id)
                    label.inst_id = e
                    label.worker_id = core.data.Worker.GOLD
                    label.val = val
                    inst = self.get_instance(e)
                    inst.add_true_label(label)
                    self.add_label_info(label_id, val)
                    count += 1
                    print(str(count) + '. inst ('+str(e)+') add label ('+str(label_id)+') true value (' + str(val)+')', end='\n')
        print(str(count) + ' true labels added', end='\n')
