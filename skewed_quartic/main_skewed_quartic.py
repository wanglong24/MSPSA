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
def norm_theta(theta, theta_0, theta_star):
    normalized_theta = np.linalg.norm(theta - theta_star.reshape(-1,1,1), axis=0) ** 2
    return np.mean(normalized_theta, 1) / np.linalg.norm(theta_0 - theta_star) ** 2

def norm_loss(loss, loss_0, loss_star):
    return np.mean((loss - loss_star) / (loss_0 - loss_star), 1)

def loss_noisy(theta):
    return loss_obj.loss(theta) + np.random.normal(0,1,1)

def loss_true(theta):
    return loss_obj.loss(theta)

p = 100; d = 50
loss_obj = SkewedQuartic(p)

theta_star = np.zeros(p)
loss_star = loss_true(theta_star)

theta_0 = np.ones(p) * 5
loss_0 = loss_true(theta_0)
np.savez("data/skewed_quartic",
         theta_star=theta_star, loss_star=loss_star,
         theta_0=theta_0, loss_0=loss_0)

### algorithm parameters ###
iter_num = 5000; rep_num = 20

MSPSA_solver = MSPSA(a=0.1, c=0.5, A=500, alpha=0.667, gamma=0.1666,
                     iter_num=int(iter_num/2), rep_num=rep_num,
                     theta_0=theta_0, d=d,
                     loss_true=loss_true, loss_noisy=loss_noisy)
MSPSA_solver.train()
MSPSA_norm_theta = norm_theta(MSPSA_solver.theta_k_all, theta_0, theta_star)
MSPSA_norm_loss = norm_loss(MSPSA_solver.loss_k_all, loss_0, loss_star)
plt.figure()
plt.plot(MSPSA_norm_theta)
plt.figure()
plt.plot(MSPSA_norm_loss)
np.savez("data/skewed_quartic_MSPSA",
         norm_theta=MSPSA_norm_theta,
         norm_loss=MSPSA_norm_loss)

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

