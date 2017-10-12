# -*- coding: utf-8 -*-

import numpy
import math
from core import data, samplable, utils
from inference import model

class MMLIWorker:

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
                    pi[i][j] = samplable.RealV(0.9)
                else:
                    pi[i][j] = samplable.RealV(0.10 / (self.K - 1))

    def m_update(self, instances, m):
        curr_pi = numpy.ndarray(shape=(self.K + 1, self.K + 1), dtype=int, order='C')
        curr_pi.fill(0)
        for inst in instances:
            d = self.worker.get_label_val_for_inst(inst.inst.id, m)
            if d != 0:
                curr_pi[inst.y_list[m].getV()][d] += 1
        for k in range(1, self.K + 1):
            s = 0
            for q in range(1, self.K + 1):
                s += curr_pi[k][q]
            if s != 0:
                for d in range(1, self.K + 1):
                    self.pi_list[m][k][d].append(float(curr_pi[k][d]) / float(s))
            else:
                for d in range(1, self.K + 1):
                    self.pi_list[m][k][d].append(self.pi_list[m][k][d].getV())

    def m_update_soft(self, instances, m):
        curr_pi = numpy.ndarray(shape=(self.K + 1, self.K + 1), dtype=int, order='C')
        curr_pi.fill(0)
        for inst in instances:
            d = self.worker.get_label_val_for_inst(inst.inst.id, m)
            if d != 0:
                for index in range(1, len(inst.y_combination)):
                    y_con = inst.y_combination[index]
                    curr_pi[y_con[m]][d] += inst.prob_list[index].getV()
        for k in range(1, self.K + 1):
            s = 0
            for q in range(1, self.K + 1):
                s += curr_pi[k][q]
            if s != 0:
                for d in range(1, self.K + 1):
                    self.pi_list[m][k][d].append(float(curr_pi[k][d]) / float(s))
            else:
                for d in range(1, self.K + 1):
                    self.pi_list[m][k][d].append(self.pi_list[m][k][d].getV())

    def print_pis(self):
        for m in range(1, self.M + 1):
            print('pi (' + str(m) + ') of worker ' + str(self.worker.id) + ':')
            for i in range(1, self.K + 1):
                for j in range(1, self.K + 1):
                    self.pi_list[m][i][j].print_obj()
                print('')


class MMLIInstance:

    def __init__(self, inst):
        self.inst = inst
        self.M = 0
        self.K = 0
        self.y_list = []
        self.prob_list = []
        self.likelihood = samplable.RealV(0.0)
        self.y_combination = []

    def initialize(self, M, K):
        self.M = M
        self.K = K
        self.y_list.append(None)
        self.y_combination.append(None)
        self.prob_list.append(None)
        for m in range(1, self.M + 1):
            self.y_list.append(samplable.IntV(0))
        self.y_combination.extend(utils.get_full_combination(self.M, self.K))
        for i in range(1, len(self.y_combination)):
            self.prob_list.append(samplable.RealV(0.0))
        #self.printYs()
        #self.printProbs()

    def compute_likelihood(self, workers, theta_list):
        prob_L = []
        for index in range(1, len(self.y_combination)):
            y_con = self.y_combination[index]
            prod_theta = 1.0
            for m in range(1, self.M + 1):
                prod_theta *= theta_list[m][y_con[m]].getV()
            prod_pi = 1.0
            for w in workers:
                for label_id in range(1, self.M + 1):
                    val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                    if val == 0:
                        continue
                    else:
                        prod_pi *= w.pi_list[label_id][y_con[label_id]][val].getV()
            prob_L.append(prod_theta * prod_pi)
        self.likelihood.append(sum(prob_L))
        return self.likelihood.getV()

    def e_step(self, workers, theta_list):
        for index in range(1, len(self.y_combination)):
            y_con = self.y_combination[index]
            prod_theta = 1.0
            for m in range(1, self.M + 1):
                prod_theta *= theta_list[m][y_con[m]].getV()
            prod_pi = 1.0
            for w in workers:
                for label_id in range(1, self.M + 1):
                    val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                    if val == 0:
                        continue
                    else:
                        prod_pi *= w.pi_list[label_id][y_con[label_id]][val].getV()
            self.prob_list[index].append(prod_theta * prod_pi)
        estimated_pos = self.compute_estimated_ys()
        for m in range(1, self.M + 1):
            self.y_list[m].append(self.y_combination[estimated_pos][m])

    def e_step_soft(self, workers, theta_list):
        for index in range(1, len(self.y_combination)):
            y_con = self.y_combination[index]
            prod_theta = 1.0
            for m in range(1, self.M + 1):
                prod_theta *= theta_list[m][y_con[m]].getV()
            prod_pi = 1.0
            for w in workers:
                for label_id in range(1, self.M + 1):
                    val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                    if val == 0:
                        continue
                    else:
                        prod_pi *= w.pi_list[label_id][y_con[label_id]][val].getV()
            self.prob_list[index].append(prod_theta * prod_pi)
        sum = 0.0
        for index in range(1,  len(self.y_combination)):
            sum +=  self.prob_list[index].getV()
        for index in range(1,  len(self.y_combination)):
            self.prob_list[index].setV(self.prob_list[index].getV()/sum)

    def compute_estimated_ys(self, random=False):
        prob = []
        prob.append(None)
        for index in range(1, len(self.prob_list)):
            prob.append(self.prob_list[index].getV())
        maxindex = utils.get_max_index(prob, random)
        return maxindex

    def final_aggregate(self):
        for m in range(1, self.M + 1):
            y_val = self.y_list[m].getV()
            # set integrated label
            integrated_label = data.Label(m)
            integrated_label.inst_id = self.inst.id
            integrated_label.worker_id = data.Worker.AGGR
            integrated_label.val = y_val
            self.inst.add_integrated_label(integrated_label)
           # print('instance ' + str(self.inst.id) + ' get integrated label (' + str(m) + ') :' + str(integrated_label.val))

    def final_aggregate_soft(self):
        estimated_pos = self.compute_estimated_ys(True)
        for m in range(1, self.M + 1):
            self.y_list[m].append(self.y_combination[estimated_pos][m])
        for m in range(1, self.M + 1):
            y_val = self.y_list[m].getV()
            # set integrated label
            integrated_label = data.Label(m)
            integrated_label.inst_id = self.inst.id
            integrated_label.worker_id = data.Worker.AGGR
            integrated_label.val = y_val
            self.inst.add_integrated_label(integrated_label)

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


class MMLIModel(model.Model):
    """
    multi-label multi-class independent model
    """
    def __init__(self, maxrnd):
        model.Model.__init__(self)
        self.workers = []
        self.instances = []
        self.I = 0
        self.J = 0
        self.K = 0
        self.M = 0  # M is the number of label ids
        self.theta_list = []
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
            mmli_inst = MMLIInstance(inst)
            mmli_inst.initialize(self.M, self.K)
            self.instances.append(mmli_inst)
        for w in range(1, self.J + 1):
            worker = dataset.get_worker(w)
            mmli_worker = MMLIWorker(worker)
            mmli_worker.initialize(self.M, self.K)
            self.workers.append(mmli_worker)
        self.theta_list.append(None)
        for m in range(1, self.M + 1):
            thetas = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
            for i in range(1, self.K + 1):
                thetas[i] = samplable.RealV(1.0 / self.K)
            self.theta_list.append(thetas)

    def initialK(self, dataset):
        maxK = 0
        for k in range(1, self.M + 1):
            currentK = dataset.get_label_val_size(k)
            if currentK > maxK:
                maxK = currentK
        self.K = maxK

    def loglikelihood(self):
        log_like = 0
        for inst in self.instances:
            l = inst.compute_likelihood(self.workers, self.theta_list)
            log_like += math.log(l)
        return log_like

    def e_step(self):
        for inst in self.instances:
            inst.e_step(self.workers, self.theta_list)

    def e_step_soft(self):
        for inst in self.instances:
            inst.e_step_soft(self.workers, self.theta_list)

    def m_step(self):
        for m in range (1, self.M + 1):
            self.m_update(m)

    def m_step_soft(self):
        for m in range(1, self.M + 1):
            self.m_update_soft(m)

    def m_update(self, m):
        # calculate theta
        numlist = []
        for k in range(0, self.K + 1):
            numlist.append(0.0)
        for inst in self.instances:
            numlist[inst.y_list[m].getV()] += 1.0
        s = sum(numlist)
        for k in range(1, self.K + 1):
            self.theta_list[m][k].append(numlist[k] / s)
        # self.print_theta()
        for w in self.workers:
            w.m_update(self.instances, m)

    def m_update_soft(self, m):
        # calculate theta
        numlist = []
        for k in range(0, self.K + 1):
            numlist.append(0.0)
        for inst in self.instances:
            for index in range(1, len(inst.y_combination)):
                y_con = inst.y_combination[index]
                numlist[y_con[m]] += inst.prob_list[index].getV()
        s = sum(numlist)
        for k in range(1, self.K + 1):
            self.theta_list[m][k].append(numlist[k] / s)
        # self.print_theta()
        for w in self.workers:
            w.m_update_soft(self.instances, m)

    def infer(self, dataset, soft=False):
        self.initialize(dataset)
        count = 1
        last_likelihood = 0
        curr_likehihood = self.loglikelihood()
        print('MMLI initial log-likelihood = ' + str(curr_likehihood))
        while ((count <= self.maxround) and (abs(curr_likehihood - last_likelihood) > model.Model.LIKELIHOOD_DIFF)):
            if (soft==False):
                self.e_step()
                self.m_step()
            else:
                self.e_step_soft()
                self.m_step_soft()
            last_likelihood = curr_likehihood
            curr_likehihood = self.loglikelihood()
            print('MMLI round (' + str(count) + ') log-likelihood = ' + str(curr_likehihood))
            count += 1
        if (soft == False):
            self.final_aggregate()
        else:
            self.final_aggregate_soft()

    def final_aggregate(self):
        for inst in self.instances:
            inst.final_aggregate()

    def final_aggregate_soft(self):
        for inst in self.instances:
            inst.final_aggregate_soft()

    def print_theta(self):
        for m in range(1, self.M + 1):
            print('probability of classes on label (' + str(m) +'):')
            for k in range(1, self.K + 1):
                self.theta_list[m][k].print_obj()
            print('')