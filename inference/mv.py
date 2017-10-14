# -*- coding: utf-8 -*-

import numpy
import core.data
import inference.model

class MVModel(inference.model.Model):
    """
    majority voting model for both single and multi-class
    """
    def __init__(self):
        inference.model.Model.__init__(self)

    def infer(self, dataset, soft=False):
        num_instance = dataset.get_instance_size()
        for inst_id in range(1, num_instance + 1):
            inst = dataset.get_instance(inst_id)
            label_ids = inst.get_label_id_list()
            for label_id in label_ids:
                labels = inst.get_noisy_labels(label_id)
                num_class = dataset.get_label_val_size(label_id)
                voted = self._vote(labels, num_class)
                # set integrated label
                integrated_label = core.data.Label(label_id)
                integrated_label.inst_id = inst.id
                integrated_label.worker_id = core.data.Worker.AGGR
                integrated_label.val = voted
                inst.add_integrated_label(integrated_label)

    def _vote(self, labels, num_class):
        counts = dict()
        for k in range(1, num_class + 1):
            counts.setdefault(k, 0)
        for label in labels:
            counts[label.val] += 1
        # find max number
        maxval = counts[1]
        maxindex = 1
        for (k, v) in counts.items():
            if v > maxval:
                maxval = v
                maxindex = k
        # find multiple max value
        maxlist = list()
        maxlist.append(maxindex)
        for (k, v) in counts.items():
            if v == maxval:
                maxlist.append(k)
        pos = numpy.random.randint(0, len(maxlist))
        return  maxlist[pos]