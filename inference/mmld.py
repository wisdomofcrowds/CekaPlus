# -*- coding: utf-8 -*-

import numpy
import math
from core import data, samplable, utils
from inference import model

class MMLDWorker:

    def __init__(self, worker):
        self.worker = worker
        self.M = 0
        self.K = 0
        self.R = 0
        self.pi_list = []

    def initialize(self, M, K, R):
        self.M = M
        self.K = K
        self.R = R
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
        self.R = 0
        self.y_list = []
        self.prob_list = []
        self.likelihood = samplable.RealV(0.0)
        self.y_combination = []
        self.z = None
        self.prob_z_list = []

    def initialize(self, M, K, R):
        self.M = M
        self.K = K
        self.R = R
        self.y_list.append(None)
        self.y_combination.append(None)
        self.prob_list.append(None)
        for m in range(1, self.M + 1):
            self.y_list.append(samplable.IntV(0))
        self.y_combination.extend(utils.get_full_combination(self.M, self.K))
        for i in range(1, len(self.y_combination)):
            self.prob_list.append(samplable.RealV(0.0))
        self.z =samplable.IntV(0)
        self.prob_z_list.append(None)
        for r in range (1, self.R + 1):
            self.prob_z_list.append(samplable.RealV(0.0))
        #self.printYs()
        #self.printProbs()

    def compute_likelihood(self, workers, theta_dict, omega):
        prob_L = []
        for index in range(1, len(self.y_combination)):
            y_con = self.y_combination[index]
            prod_theta = [0.0]
            prod_pi = [0.0]
            theta_pi = [0.0]
            for r in range(1, self.R + 1):
                prod_theta.append(1.0)
                prod_pi.append(1.0)
                theta_pi.append(1.0)
            for r in range (1, self.R + 1):
                theta_list = theta_dict.get(r)
                for m in range(1, self.M + 1):
                    prod_theta[r] *= theta_list[m][y_con[m]].getV()
                for w in workers:
                    for label_id in range(1, self.M + 1):
                        val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                        if val == 0:
                            continue
                        else:
                            prod_pi[r] *= w.pi_list[label_id][y_con[label_id]][val].getV()
                theta_pi[r] *= (prod_theta[r] * prod_pi[r] * omega[r].getV())
            prob_L.append(sum(theta_pi))
        self.likelihood.append(sum(prob_L))
        return self.likelihood.getV()

    def e_step(self, workers, theta_dict, omega):
        prob_z = [None]
        for r in range (1, self.R + 1):
            prob_z.append(0.0)
        for index in range(1, len(self.y_combination)):
            y_con = self.y_combination[index]
            prod_theta = [0.0]
            prod_pi = [0.0]
            theta_pi = [0.0]
            for r in range(1, self.R + 1):
                prod_theta.append(1.0)
                prod_pi.append(1.0)
                theta_pi.append(1.0)
            for r in range(1, self.R + 1):
                theta_list = theta_dict.get(r)
                for m in range(1, self.M + 1):
                    prod_theta[r] *= theta_list[m][y_con[m]].getV()
                for w in workers:
                    for label_id in range(1, self.M + 1):
                        val = w.worker.get_label_val_for_inst(self.inst.id, label_id)
                        if val == 0:
                            continue
                        else:
                            prod_pi[r] *= w.pi_list[label_id][y_con[label_id]][val].getV()
                theta_pi[r] *= (prod_theta[r] * prod_pi[r] * omega[r].getV())
                prob_z[r] += theta_pi[r]
            self.prob_list[index].append(sum(theta_pi))
        for r in range(1, self.R + 1):
            self.prob_z_list[r].append(prob_z[r])
        estimated_y_pos = self.compute_estimated_ys()
        for m in range(1, self.M + 1):
            self.y_list[m].append(self.y_combination[estimated_y_pos][m])
        self.z.append(self.compute_estimated_z())

    def compute_estimated_ys(self):
        prob = []
        prob.append(None)
        for index in range(1, len(self.prob_list)):
            prob.append(self.prob_list[index].getV())
        maxindex = utils.get_max_index(prob)
        return maxindex

    def compute_estimated_z(self):
        prob = []
        prob.append(None)
        for index in range(1, len(self.prob_z_list)):
            prob.append(self.prob_z_list[index].getV())
        maxindex = utils.get_max_index(prob)
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
            print('instance ' + str(self.inst.id) + ' get integrated label (' + str(m) + ') :' + str(integrated_label.val))

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
    def __init__(self, R, maxrnd):
        model.Model.__init__(self)
        self.workers = []
        self.instances = []
        self.I = 0
        self.J = 0
        self.K = 0
        self.M = 0  # M is the number of label ids
        self.R = R
        self.theta_dict = dict()
        self.maxround = maxrnd
        self.omega = [None]

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
            mmld_inst.initialize(self.M, self.K, self.R)
            self.instances.append(mmld_inst)
        for w in range(1, self.J + 1):
            worker = dataset.get_worker(w)
            mmld_worker = MMLDWorker(worker)
            mmld_worker.initialize(self.M, self.K, self.R)
            self.workers.append(mmld_worker)
        for r in  range(1, self.R + 1):
            theta_list = list()
            theta_list.append(None)
            for m in range(1, self.M + 1):
                thetas = numpy.ndarray(shape=(self.K + 1), dtype=samplable.RealV, order='C')
                for i in range(1, self.K + 1):
                    thetas[i] = samplable.RealV(1.0 / self.K)
                theta_list.append(thetas)
            self.theta_dict.setdefault(r, theta_list)
            # initialize omega
            if (len(self.omega)==1):
                self.omega.append(samplable.RealV(1.0 / self.R))

    def set_omega(self, vals):
        for r in range(1, self.R + 1):
            self.omega.append(samplable.RealV(vals[r]))

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
            l = inst.compute_likelihood(self.workers, self.theta_dict, self.omega)
            log_like += math.log(l)
        return log_like

    def e_step(self):
        for inst in self.instances:
            inst.e_step(self.workers, self.theta_dict, self.omega)

    def m_step(self):
        for m in range(1, self.M + 1):
            self.m_update(m)

    def m_update(self, m):
        # calculate omega
        num_inst = []
        for r in range(0, self.R + 1):
            num_inst.append(0.0)
        for inst in self.instances:
            num_inst[inst.z.getV()] += 1.0
        s = sum(num_inst)
        for r in range(1, self.R + 1):
            self.omega[r].append(num_inst[r] / s)
        # calculate theta
        for r in range(1, self.R + 1):
            numlist = []
            for k in range(0, self.K + 1):
                numlist.append(0.0)
            for inst in self.instances:
                if inst.z.getV() == r:
                    numlist[inst.y_list[m].getV()] += 1.0
            ss = sum(numlist)
            for k in range(1, self.K + 1):
                if (ss == 0):
                    self.theta_dict.get(r)[m][k].append(0.0)
                else:
                    self.theta_dict.get(r)[m][k].append(numlist[k] / ss)
        # self.print_theta()
        for w in self.workers:
            w.m_update(self.instances, m)

    def infer(self, dataset):
        self.initialize(dataset)
        count = 1
        last_likelihood = 0
        curr_likehihood = self.loglikelihood()
        print('MMLD initial log-likelihood = ' + str(curr_likehihood))
        while ((count <= self.maxround) and (abs(curr_likehihood - last_likelihood) > model.Model.LIKELIHOOD_DIFF)):
            self.e_step()
            self.m_step()
            last_likelihood = curr_likehihood
            curr_likehihood = self.loglikelihood()
            print('MMLD round (' + str(count) + ') log-likelihood = ' + str(curr_likehihood))
            count += 1
        self.final_aggregate()

    def final_aggregate(self):
        for inst in self.instances:
            inst.final_aggregate()

    def print_theta(self):
        for r in range(1, self.R+1):
            for m in range(1, self.M + 1):
                print('probability of classes on label (' + str(m) +') in r=' + str(r) +' branch: ')
                for k in range(1, self.K + 1):
                    self.theta_dict.get(r)[m][k].print_obj()
                print('')