#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 01:41:35 2020

@author: longwang
"""

import numpy as np

class StochasticRuler:
    def __init__(self,
                 iter_num=100, rep_num=1, direct_num=1,
                 theta_0=None, d=0, loss_true=None, loss_noisy=None,
                 record_theta_flag=True, record_loss_flag=True):

        # sigma: new candidate point theta + normal(0, simga^2)
        # direct_num: number of directions per iteration
        # d: the first d components are integers

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.direct_num = direct_num

        self.theta_0 = theta_0
        self.loss_0 = loss_true(theta_0)

        self.p = theta_0.shape[0] # shape = (p,)
        self.d = d

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

    def project(self, theta):
        # project the first d components to the nearest integer
        proj_theta = theta.copy()
        proj_theta[:self.d] = np.round(proj_theta[:self.d])
        return proj_theta

    def train(self):
        if self.record_theta_flag:
            self.theta_k_all = np.empty((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_k_all = np.empty((self.iter_num, self.rep_num))

        for rep_idx in range(self.rep_num):
            print("Algo: stochastic ruler, rep_idx:", rep_idx, "/", self.rep_num)
            theta = self.theta_0

            for iter_idx in range(self.iter_num):
                M_k = int(np.ceil(0.5 * np.log(iter_idx + 2)))
                theta_new = np.random.rand(self.p) * self.theta_0
                theta_new = self.project(theta_new)
                accept_flag = True
                for i in range(M_k):
                    loss_new = self.loss_noisy(theta_new)
                    if loss_new > self.loss_0 * np.random.rand(1):
                        accept_flag = False
                        break

                if accept_flag:
                    theta = theta_new

                # record result
                if self.record_theta_flag:
                    self.theta_k_all[:,iter_idx,rep_idx] = self.project(theta)
                if self.record_loss_flag:
                    self.loss_k_all[iter_idx,rep_idx] = self.loss_true(self.project(theta))