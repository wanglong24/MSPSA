#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:58:35 2020

@author: Long Wang
"""

import numpy as np

class MSPSA:
    def __init__(self, a=0, c=0.1, A=0, alpha=0.602, gamma=0.101,
                 iter_num=100, rep_num=1, direct_num=1,
                 theta_0=None, d=0, lb=None, ub=None,
                 loss_true=None, loss_noisy=None,
                 record_theta_flag=True, record_loss_flag=True,
                 seed=1,
                 update_loss=None):

        # step size: a_k = a / (k+1+A) ** alpha
        # perturbation size: c_k = c / (k+1) ** gamma
        # direct_num: number of directions per iteration
        # d: the first d components are integers
        # lb/ub: lower/upper bound of the variable List[p]

        self.seed = seed
        np.random.seed(self.seed)

        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.direct_num = direct_num

        self.theta_0 = theta_0
        self.p = theta_0.shape[0] # shape = (p,)
        self.d = d
        # initialize perturbation vector: uniform{-1, 1}
        self.delta_all = np.round(np.random.rand(self.p, direct_num, iter_num, rep_num)) * 2 - 1

        if lb == None:
            self.lb = -np.ones(self.p) * np.Inf
        else:
            self.lb = lb
        if ub == None:
            self.ub = np.ones(self.p) * np.Inf
        else:
            self.ub = ub

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy
        self.update_loss = update_loss

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

    def pi(self, theta):
        # project the first d components to floor(x) + 0.5
        pi_theta = theta.copy()
        pi_theta[:self.d] = np.floor(pi_theta[:self.d]) + 0.5
        return pi_theta

    def project(self, theta):
        # project the first d components to the nearest integer
        proj_theta = theta.copy()
        proj_theta[:self.d] = np.round(proj_theta[:self.d])
        proj_theta = np.clip(proj_theta, self.lb, self.ub)
        return proj_theta

    def get_grad_est(self, theta, iter_idx=0, rep_idx=0):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        C_k = np.concatenate((np.repeat(0.5, self.d), np.repeat(c_k, self.p-self.d)))

        grad_all = np.empty((self.p, self.direct_num))
        for direct_idx in range(self.direct_num):
            delta = self.delta_all[:, direct_idx, iter_idx, rep_idx]
            loss_plus = self.loss_noisy(self.pi(theta) + C_k * delta)
            loss_minus = self.loss_noisy(self.pi(theta) - C_k * delta)
            grad_all[:,direct_idx] = (loss_plus - loss_minus) / (2 * C_k * delta)

        grad = np.average(grad_all, axis=1)
        return grad

    def get_new_est(self, theta, grad, iter_idx=0):
        a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
        a_k = a_k * np.array([1,1,10,10])
        return theta - a_k * grad

    def train(self):
        if self.record_theta_flag:
            self.theta_k_all = np.empty((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_k_all = np.empty((self.iter_num, self.rep_num))

        for rep_idx in range(self.rep_num):
            print("Algo: MSPSA, rep_idx:", rep_idx+1, "/", self.rep_num)
            theta = self.theta_0

            for iter_idx in range(self.iter_num):
                grad = self.get_grad_est(theta, iter_idx, rep_idx)
                theta = self.get_new_est(theta, grad, iter_idx)

                self.update_loss(theta, iter_idx)

                # record result
                if self.record_theta_flag:
                    self.theta_k_all[:,iter_idx,rep_idx] = self.project(theta)
                if self.record_loss_flag:
                    self.loss_k_all[iter_idx,rep_idx] = self.loss_true(self.project(theta))