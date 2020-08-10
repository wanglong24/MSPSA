#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:07:04 2020

@author: Long Wang
"""

import numpy as np

class RandomSearch:
    def __init__(self, sigma=0.1,
                 iter_num=100, rep_num=1,
                 theta_0=None, loss_true=None, loss_noisy=None,
                 d=0,
                 record_theta_flag=True, record_loss_flag=True,
                 seed=1):

        # sigma: new candidate point theta + normal(0, simga^2)
        # d: the first d components are integers

        self.seed = seed
        np.random.seed(self.seed)

        self.sigma = sigma

        self.iter_num = iter_num
        self.rep_num = rep_num

        self.theta_0 = theta_0
        self.p = theta_0.shape[0] # shape = (p,)
        self.delta_all = np.random.normal(0, sigma, (self.p, iter_num, rep_num))

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        self.d = d

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
            print("algo: Random Search; rep_idx:", rep_idx+1, "/", self.rep_num)
            theta = self.theta_0
            loss_best = np.Inf

            for iter_idx in range(self.iter_num):
                theta_new = self.project(theta + self.delta_all[:, iter_idx, rep_idx])
                loss_new = self.loss_noisy(theta_new)
                if loss_new < loss_best:
                    theta = theta_new
                    loss_best = loss_new

                # record result
                if self.record_theta_flag:
                    self.theta_k_all[:,iter_idx,rep_idx] = self.project(theta)
                if self.record_loss_flag:
                    self.loss_k_all[iter_idx,rep_idx] = self.loss_true(self.project(theta))