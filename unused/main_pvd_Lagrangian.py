#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:04:04 2020

@author: Long Wang
"""

import numpy as np
import matplotlib.pyplot as plt

from loss.pressure_vessel_design_Lagrange import PressureVesselDesignLagrange

from algorithm.mspsa import MSPSA
from algorithm.random_search import RandomSearch
from algorithm.stochastic_ruler import StochasticRuler

### normalize result ###
def norm_theta(theta, theta_0, theta_star):
    normalized_theta = np.linalg.norm(theta - theta_star.reshape(-1,1,1), axis=0) ** 2
    return np.mean(normalized_theta, 1) / np.linalg.norm(theta_0 - theta_star) ** 2

def norm_loss(loss, loss_0, loss_star):
    return np.mean((loss - loss_star) / (loss_0 - loss_star), 1)

def loss_noisy(theta):
    return loss_obj.loss_Lagrangian(theta)# + np.random.normal(0,100,1)

def loss_true(theta):
    return loss_obj.loss(theta)

def update_loss_obj(theta, iter_idx):
    if (iter_idx % 100) == 0:
        loss_obj.update_Lagrangian(theta)
    # if (iter_idx % 1000) == 0:
    #     print("iter_idx:", iter_idx, "r:", np.around([loss_obj.r_1,loss_obj.r_2,loss_obj.r_3,loss_obj.r_4],5))


p = 4; d = 2
loss_obj = PressureVesselDesignLagrange(multiplier=1.2, r=1e-2*np.ones(4), lamb=0)

theta_star = np.array([13, 7, 42.098445, 176.636596])
# 0.8125 / 0.0625 = 13
# 0.4375 / 0.0625 = 7

loss_star = loss_true(theta_star) # 6059

# theta_0 = np.array([10, 10, 50, 150])
theta_0 = np.array([18, 10, 50, 150])

loss_0 = loss_true(theta_0)
np.savez("data/pvd",
         theta_star=theta_star, loss_star=loss_star,
         theta_0=theta_0, loss_0=loss_0)

### algorithm parameters ###
iter_num = 100000; rep_num = 1

# a = 0.0001, c = 0.5

MSPSA_solver = MSPSA(a=0.00001, c=1, A=500, alpha=0.667, gamma=0.1666,
                     iter_num=int(iter_num/2), rep_num=rep_num, direct_num=1,
                     theta_0=theta_0, d=d, lb=[1,1,10,10], ub=[99,99,200,200],
                     loss_true=loss_true, loss_noisy=loss_noisy,
                     update_loss=update_loss_obj, seed=1)
MSPSA_solver.train()
# MSPSA_norm_theta = norm_theta(MSPSA_solver.theta_k_all, theta_0, theta_star)
# MSPSA_norm_loss = norm_loss(MSPSA_solver.loss_k_all, loss_0, loss_star)

MSPSA_norm_theta = np.mean(MSPSA_solver.theta_k_all, 1)
MSPSA_norm_loss = np.mean(MSPSA_solver.loss_k_all, 1)

loss_obj.show_g(MSPSA_solver.theta_k_all[:,-1])
print("theta:", MSPSA_solver.theta_k_all[:,-1].T)
print("loss:", MSPSA_norm_loss[-1])

# plt.figure()
# plt.plot(MSPSA_norm_theta)
plt.figure()
plt.plot(MSPSA_norm_loss)

# np.savez("data/skewed_quartic_MSPSA",
#          norm_theta=MSPSA_norm_theta,
#          norm_loss=MSPSA_norm_loss)

# RS_solver = RandomSearch(sigma=0.05,
#                          iter_num=iter_num, rep_num=rep_num,
#                          theta_0=theta_0, d=d,
#                          loss_true=loss_true, loss_noisy=loss_noisy)
# RS_solver.train()
# RS_norm_theta = norm_theta(RS_solver.theta_k_all, theta_0, theta_star)
# RS_norm_loss = norm_loss(RS_solver.loss_k_all, loss_0, loss_star)
# plt.figure()
# plt.title("theta")
# plt.plot(RS_norm_theta)
# plt.figure()
# plt.title("loss")
# plt.plot(RS_norm_loss)
# np.savez("data/skewed_quartic_RS",
#          norm_theta=RS_norm_theta,
#          norm_loss=RS_norm_loss)

# SR_solver = StochasticRuler(iter_num=iter_num, rep_num=rep_num,
#                         theta_0=theta_0, d=d,
#                         loss_true=loss_true, loss_noisy=loss_noisy)
# SR_solver.train()
# SR_norm_theta = norm_theta(SR_solver.theta_k_all, theta_0, theta_star)
# SR_norm_loss = norm_loss(SR_solver.loss_k_all, loss_0, loss_star)
# plt.figure()
# plt.title("theta")
# plt.plot(SR_norm_theta)
# plt.figure()
# plt.title("loss")
# plt.plot(SR_norm_loss)
# np.savez("data/skewed_quartic_SR",
#          norm_theta=SR_norm_theta,
#          norm_loss=SR_norm_loss)