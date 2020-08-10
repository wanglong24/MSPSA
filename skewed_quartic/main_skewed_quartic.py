#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:22:09 2020

@author: Long Wang
"""

import numpy as np
import matplotlib.pyplot as plt

from loss.skewed_quartic import SkewedQuartic

from algorithm.mspsa import MSPSA
from algorithm.random_search import RandomSearch
from algorithm.stochastic_ruler import StochasticRuler

### normalize result ###
def loss_noisy(theta):
    return loss_obj.loss(theta) + np.random.normal(0,1,1)

def loss_true(theta):
    return loss_obj.loss(theta)

def get_theta_norm_error(theta, theta_0, theta_star):
    normalized_theta = np.linalg.norm(theta - theta_star.reshape(-1,1,1), axis=0)
    return np.mean(normalized_theta, 1) / np.linalg.norm(theta_0 - theta_star)

def get_loss_norm_error(loss, loss_0, loss_star):
    return np.mean((loss - loss_star) / (loss_0 - loss_star), 1)

def show_result(solver, theta_0, theta_star, file_name):
    solver_theta_norm_error = get_theta_norm_error(solver.theta_k_all, theta_0, theta_star)
    solver_loss_norm_error = get_loss_norm_error(solver.loss_k_all, loss_0, loss_star)

    plt.figure()
    plt.title("theta normalized error")
    plt.plot(solver_theta_norm_error)
    plt.figure()
    plt.title("loss normalized error")
    plt.plot(solver_loss_norm_error)

    np.savez("data/" + file_name,
             theta_norm_error=solver_theta_norm_error,
             loss_norm_error=solver_loss_norm_error)

p = 100; d = 50
loss_obj = SkewedQuartic(p)

theta_star = np.zeros(p); loss_star = loss_true(theta_star)
theta_0 = np.ones(p) * 5; loss_0 = loss_true(theta_0)
np.savez("data/skewed-quartic",
         theta_star=theta_star, loss_star=loss_star,
         theta_0=theta_0, loss_0=loss_0)

### algorithm parameters ###
meas_num = 5000; rep_num = 20

### MSPSA ###
MSPSA_solver = MSPSA(a=0.1, c=0.5, A=500, alpha=0.667, gamma=0.1666,
                     iter_num=int(meas_num/2), rep_num=rep_num,
                     theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                     d=d, seed=1)
MSPSA_solver.train()
show_result(MSPSA_solver, theta_0, theta_star, "skewed-quartic-MSPSA")

### Random Search ###
RS_solver = RandomSearch(sigma=0.05,
                          iter_num=meas_num, rep_num=rep_num,
                          theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                          d=d, seed=1)
RS_solver.train()
show_result(RS_solver, theta_0, theta_star, "skewed-quartic-RS")

### Stochastic Ruler ###
M_multiplier = 0.5
SR_meas_num = 0
SR_iter_num = 0
SR_iter_seq = []
while SR_meas_num < meas_num:
    M_k = int(np.ceil(M_multiplier * np.log(SR_iter_num + 2)))
    SR_meas_num += M_k
    SR_iter_seq.append(M_k)
    SR_iter_num += 1

SR_solver = StochasticRuler(M_multiplier=M_multiplier,
                            iter_num=SR_iter_num, rep_num=rep_num,
                            theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                            d=d, seed=1)
SR_solver.train()
show_result(SR_solver, theta_0, theta_star, "skewed-quartic-SR")
np.save("data/skewed-quartic-SR-iter-seq", SR_iter_seq)