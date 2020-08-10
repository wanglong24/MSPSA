#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 00:10:19 2020

@author: Long Wang
"""

import numpy as np

class GradDSLI:
    def __init__(self,
                 iter_num=100, rep_num=1, direct_num=1,
                 theta_0=None, d=0, loss_true=None, loss_noisy=None,
                 record_theta_flag=True, record_loss_flag=True):

        # d: the first d components are integers

        self.m = 2
        self.delta = 2

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.direct_num = direct_num

        self.theta_0 = theta_0
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

    def DSLI(self, theta):
        # get the linear interpolation and the gradient
        f_LI = 0 # linear interpolation
        g_LI = np.empty(self.p) # gradient for the linear interpolation

        Delta = np.ones(self.p)
        Delta[self.d:] = self.delta

        x = theta
        y = np.floor(x / Delta) * Delta
        z = (x - y) / Delta
        z = np.concatenate(([0],z,[1]))
        z_arg = np.argsort(-z) # sort in decreasing order

        x_new = np.empty((self.p, self.p + 1))
        x_new[:,0] = y
        for i in range(1, self.p + 1):
            x_new[:,i] = x_new[:,i-1].copy()
            x_new[z_arg[i]-1,i] += Delta[z_arg[i]-1]

        f_new = np.zeros(self.p + 1)
        for i in range(self.p + 1):
            w_i = z[z_arg[i]] - z[z_arg[i+1]]
            for j in range(self.m):
                f_new[i] += self.loss_noisy(x_new[:,i]) / self.m
            f_LI += w_i * f_new[i]

        for i in range(1, self.p + 1):
            g_LI[i-1] = (f_new[i] - f_new[i-1]) / Delta[z_arg[i]-1]

        return f_LI, g_LI

    def train(self):
        if self.record_theta_flag:
            self.theta_k_all = np.empty((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_k_all = np.empty((self.iter_num, self.rep_num))

        for rep_idx in range(self.rep_num):
            print("rep_idx:", rep_idx, "/", self.rep_num)
            theta = self.theta_0

            for iter_idx in range(self.iter_num):
                self.m = int(np.ceil(self.m * 1.5))
                self.delta = self.delta / 2


                f_LI, g_LI = self.DSLI(theta)
                s = 1 ** iter_idx * 0.1
                theta_new = theta - s * g_LI / np.linalg.norm(g_LI)
                theta_new = self.project(theta)

                # record result
                if self.record_theta_flag:
                    self.theta_k_all[:,iter_idx,rep_idx] = self.project(theta)
                if self.record_loss_flag:
                    self.loss_k_all[iter_idx,rep_idx] = self.loss_true(self.project(theta))