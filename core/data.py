# -*- coding: utf-8 -*-
class Label:
    """
    The base class of Label
    """
    INVALID_ID = 0
    INVALID_VAL = -1
    SINGLE_LABLE = 3 # because there are 3 items in a line in a response file
    MULTI_LABEL = 4

    _next_id_val = {1:1} # the first term is label id, the second one is its value
    _id_name_dict = dict() # mapping label_id and label name
    _val_name_dicts = {1:dict()}  # the first term is label id, the second one is a dictionary to map values

    @classmethod
    def fetch_val_id_by_name(cls, label_id, val_name):
        val_dict = cls._val_name_dicts[label_id]
        for (k, v) in val_dict.items():
            if v == val_name:
                return k
        val_id = cls._next_id_val[label_id]
        cls._val_name_dicts[label_id].setdefault(val_id, val_name)
        cls._next_id_val[label_id] += 1
        return val_id

    @classmethod
    def fetch_id_by_name(cls, label_name):
        max_label_id = 0
        for (k, v) in cls._id_name_dict.items():
            if k > max_label_id:
                max_label_id = k
            if v == label_name:
                return k
        max_label_id += 1
        if cls._next_id_val.get(max_label_id) == None:
            cls._next_id_val.setdefault(max_label_id, 1)
        cls._id_name_dict.setdefault(max_label_id, label_name)
        return max_label_id

    @classmethod
    def get_id_by_name(cls, label_name):
        for (k, v) in cls._id_name_dict.items():
            if v == label_name:
                return k
        return cls.INVALID_ID

    @classmethod
    def get_label_val_num(cls, label_id = 1):
        return len(cls._val_name_dicts[label_id])

    @classmethod
    def get_label_id_num(cls):
        return len(cls._id_name_dict)

    def __init__(self, id = 0):
        self.id = id
        self.val = 0
        self.inst_id = 0
        self.worker_id = 0


class Worker:
    """
    The base class of crowdsourced workers
    """
    INVALID_ID = 0
    GOLD = 9999999 # gold worker that provides the ground truth
    AGGR = 9999998 # aggregated worker

    _next_id = 1 # the worker id starts from 1
    _id_name_dict = dict() # mapping worker_id and worker name

    @classmethod
    def fetch_id_by_name(cls, name):
        for (k, v) in cls._id_name_dict.items():
            if v == name:
                return k
        id = cls._next_id
        cls._id_name_dict.setdefault(id, name)
        cls._next_id += 1
        return id

    def __init__(self, id = 0):
        self.id = id
        self.label_dict = dict()
        self.sorted_label_dicts = {1:dict()}

    def get_label(self, label_info):
        if label_info in self.label_dict:
            return self.label_dict[label_info]
        return None

    def add_label(self, label):
        tpl = (label.id, label.inst_id, label.worker_id)
        if tpl not in self.label_dict:
            self.label_dict.setdefault(tpl, label)
        target_dict = self.sorted_label_dicts.get(label.id)
        if target_dict == None:
            target_dict = dict()
            self.sorted_label_dicts.setdefault(label.id, target_dict)
        target_dict.setdefault(tpl, label)

    def get_sorted_label_dict_size(self):
        return len(self.sorted_label_dicts)

    def get_label_list(self):
        list = []
        for (k, v) in self.label_dict.items():
            list.append(v)
        return list


class Instance:
    """
    The base class of instance
    """
    INVALID_ID = 0
    _next_id = 1 # the instance id starts from 1
    _id_name_dict = dict()

    @classmethod
    def fetch_id_by_name(cls, name):
        for (k, v) in cls._id_name_dict.items():
            if v == name:
                return k
        id = cls._next_id
        cls._id_name_dict.setdefault(id, name)
        cls._next_id += 1
        return id

    @classmethod
    def get_id_by_name(cls, name):
        for (k, v) in cls._id_name_dict.items():
            if v == name:
                return k
        return cls.INVALID_ID

    def __init__(self, id = 0):
        self.id = id
        self.true_label_dict = dict()
        self.intg_label_dict = dict()
        self.sorted_label_dicts = {1:list()}

    def get_true_label(self, label_id):
        return self.true_label_dict.get(label_id)

    def add_true_label(self, label):
        if label.id not in self.true_label_dict:
            self.true_label_dict.setdefault(label.id, label)

    def get_true_label_set(self):
        return self.true_label_dict

    def add_noisy_label(self, label):
        if self.sorted_label_dicts.get(label.id) == None:
            self.sorted_label_dicts.setdefault(label.id, list())
            self.sorted_label_dicts[label.id].append(label)
            return
        for l in self.sorted_label_dicts[label.id]:
            if (l.id == label.id) and (l.worker_id == label.worker_id) and (l.inst_id == label.inst_id)\
                and (l.val == label.val):
                return
        self.sorted_label_dicts[label.id].append(label)

    def get_noisy_labels(self, label_id = 1):
        return  self.sorted_label_dicts[label_id]

    def add_integrated_label(self, label):
        if label.id in self.intg_label_dict:
            self.intg_label_dict[label.id] = label
        else:
            self.intg_label_dict.setdefault(label.id, label)

    def get_integrated_label(self, label_id = 1):
        if label_id in self.intg_label_dict:
            return self.intg_label_dict[label_id]
        return None

    def get_label_id_list(self):
        id_list =[]
        for (k, v) in self.sorted_label_dicts.items():
            id_list.append(k)
        return id_list

    def equal_integrated_true(self, label_id = 1):
        if self.true_label_dict[label_id].val == self.intg_label_dict[label_id].val:
            return 1
        return 0

class Dataset:
    """
    The base class of a data set
    """
    def __init__(self):
        self.inst_dict = dict()
        self.worker_dict = dict()

    def get_instance(self, id):
        return self.inst_dict.get(id)

    def get_worker(self, id):
        return self.worker_dict.get(id)

    def add_instance(self, inst):
        if self.get_instance(inst.id) == None:
            self.inst_dict.setdefault(inst.id, inst)

    def add_worker(self, worker):
         if self.get_worker(worker.id) == None:
             self.worker_dict.setdefault(worker.id, worker)

    def get_instance_size(self):
        return len(self.inst_dict)

    def get_worker_size(self):
        return len(self.worker_dict)

    def is_multi_class(self):
        for (k, v) in self.worker_dict.items():
            if v.get_sorted_label_dict_size() > 1:
                return True
        return False