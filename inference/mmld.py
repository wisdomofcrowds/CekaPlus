# -*- coding: utf-8 -*-

import numpy
from core import data, samplable
from inference import model

class MMLDWorker:

    def __init__(self, worker):
        self.worker = worker
        self.M = 0
        self.K = 0
        self.pi_list = []

    def initialize(self, M, K):
        self.M = M
        self.K = K
        self.pi_list.append(None)
        for m in range(1, self.M + 1):
            pi = numpy.ndarray(shape=(self.K + 1, self.K + 1), dtype=samplable.RealV, order='C')
            self.initialize_pi(pi)
            self.pi_list.append(pi)
        #self.print_pis()

    def initialize_pi(self, pi):
        """
        the basic initialization of pi
        :param K:
        :param pi:
        :return:
        """
        for i in range(1, self.K + 1):
            for j in range(1, self.K + 1):
                if i == j:
                    pi[i][j] = samplable.RealV(0.8)
                else:
                    pi[i][j] = samplable.RealV(0.20 / (self.K - 1))

    def print_pis(self):
        for m in range(1, self.M + 1):
            print('pi (' + str(m) + ') of worker ' + str(self.worker.id) + ':')
            for i in range(1, self.K + 1):
                for j in range(1, self.K + 1):
                    self.pi_list[m][i][j].print_obj()
                print('')

class MMLDInstance:

    def __init__(self, inst):
        self.inst = inst
        self.M = 0
        self.K = 0
        self.y_list = []
        self.prob_list = []

    def initialize(self, M, K):
        self.M = M
        self.K = K
        self.y_list.append(None)
        self.prob_list.append(None)
        for m in range(1, self.M + 1):
            self.y_list.append(samplable.IntV(0))
            probs = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
            for i in range(1, self.K + 1):
                probs[i] = samplable.RealV(1.0 / self.K)
            self.prob_list.append(probs)
        #self.printYs()
        #self.printProbs()

    def printYs(self):
        print('The estimated class of instance ' + str(self.inst.id) +  ':', end=' ')
        for m in range(1, self.M+1):
            self.y_list[m].print_obj()
        print('')

    def printProbs(self):
        print('The probabilities of classes of instance ' + str(self.inst.id) + ' >>>')
        for m in range(1, self.M + 1):
            print('label (' + str(m)+ '):', end=' ')
            for k in range(1, self.K + 1):
                self.prob_list[m][k].print_obj()
            print('')


class MMLDModel(model.Model):
    """
    multi-label multi-class dependent model
    """
    def __init__(self, r, maxrnd):
        model.Model.__init__(self)
        self.workers = []
        self.instances = []
        self.I = 0
        self.J = 0
        self.K = 0
        self.M = 0  # M is the number of label ids
        self.r = r
        self.theta_dict = dict()
        self.maxround = maxrnd

    def initialize(self, dataset):
        self.M = dataset.get_label_id_size()
        print('The number of label ids = ' + str(self.M))
        self.I = dataset.get_instance_size()
        self.J = dataset.get_worker_size()
        print('Total ' + str(self.I) + ' instances and ' + str(self.J) + ' workers')
        self.initialK(dataset)
        print('K = ' + str(self.K))
        for i in range(1, self.I + 1):
            inst = dataset.get_instance(i)
            mmld_inst = MMLDInstance(inst)
            mmld_inst.initialize(self.M, self.K)
            self.instances.append(mmld_inst)
        for w in range(1, self.J + 1):
            worker = dataset.get_worker(w)
            mmld_worker = MMLDWorker(worker)
            mmld_worker.initialize(self.M, self.K)
            self.workers.append(mmld_worker)
        for r in  range(1, self.r + 1):
            theta_list = list()
            theta_list.append(None)
            for m in range(1, self.M + 1):
                thetas = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
                for i in range(1, self.K + 1):
                    thetas[i] = samplable.RealV(1.0 / self.K)
                theta_list.append(thetas)
            self.theta_dict.setdefault(r, theta_list)

    def initialK(self, dataset):
        maxK = 0
        for k in range(1, self.M + 1):
            currentK = dataset.get_label_val_size(k)
            if currentK > maxK:
                maxK = currentK
        self.K = maxK

    def infer(self, dataset):
        self.initialize(dataset)
        pass

