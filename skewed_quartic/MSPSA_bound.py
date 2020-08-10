#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:52:06 2020

@author: Long Wang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 12})

from loss.skewed_quartic import SkewedQuartic

from algorithm.mspsa import MSPSA

### normalize result ###
def norm_theta(theta, theta_0, theta_star):
    normalized_theta = np.linalg.norm(theta - theta_star.reshape(-1,1,1), axis=0)
    return np.mean(normalized_theta, 1) #/ np.linalg.norm(theta_0 - theta_star) ** 2

def norm_loss(loss, loss_0, loss_star):
    return np.mean((loss - loss_star) / (loss_0 - loss_star), 1)

def loss_noisy(theta):
    return loss_obj.loss(theta) + np.random.normal(0,1,1)

def loss_true(theta):
    return loss_obj.loss(theta)

p = 10; d = 5
loss_obj = SkewedQuartic(p)

theta_star = np.zeros(p)
loss_star = loss_true(theta_star)

theta_0 = np.ones(p) * 1
loss_0 = loss_true(theta_0)
np.savez("data/skewed_quartic",
         theta_star=theta_star, loss_star=loss_star,
         theta_0=theta_0, loss_0=loss_0)

### algorithm parameters ###
iter_num = 5000; rep_num = 20
a = 0.05
A = 100
alpha = 0.667
c = 0.5
gamma = 0.1666

MSPSA_solver = MSPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                     iter_num=iter_num, rep_num=rep_num,
                     theta_0=theta_0, d=d,
                     loss_true=loss_true, loss_noisy=loss_noisy)
MSPSA_solver.train()
MSPSA_norm_theta = norm_theta(MSPSA_solver.theta_k_all, theta_0, theta_star)

norm_theta_0 = np.linalg.norm(theta_0 - theta_star)
MSPSA_norm_theta = np.concatenate(([norm_theta_0**2], MSPSA_norm_theta**2))


### compute finite sample bound ###
kappa_0 = 1; kappa_1 = 1; kappa_2 = 1
sigma2_epsilon = 2
lamb = 1.1
B_H = 0.8; B_T = 0.8
sigma2_L = 15

P_0 = 1
P_1 = np.exp((2*lamb-1)*a/(1-alpha) * ((1+A)**(1-alpha) - (1+1+A)**(1-alpha)))

a_0 = a / (1+A)**alpha
a_1 = a / (1+A+1)**alpha
c_0 = c
c_1 = c / (1+1)**gamma

U_norm2 = (B_H * (p-d)**2 * kappa_0**2)**2 * d + (1/6*B_T*(
    ((p-d)**3 - (p-d-1)**3)*kappa_0**2 + (p-d-1)**3*kappa_0**3*kappa_1))**2 * (p-d)

MSPSA_norm_theta_bound = np.empty(iter_num)

for iter_idx in np.arange(iter_num):
    a_k = a / (1 + A + iter_idx) ** alpha
    c_k = c / (1 + iter_idx) ** gamma

    if iter_idx == 0:
        MSPSA_norm_theta_bound[iter_idx] = (1-(2*lamb-1)*a_k) * norm_theta_0**2 \
            + U_norm2 * a_k * c_k**4 + (sigma2_L+sigma2_epsilon) * a_k**2 * (d + (p-d)) * kappa_2 / (4 * c_k**2)
    else:
        MSPSA_norm_theta_bound[iter_idx] = (1-(2*lamb-1)*a_k) * MSPSA_norm_theta_bound[iter_idx-1] \
            + U_norm2 * a_k * c_k**4 + (sigma2_L+sigma2_epsilon) * a_k**2 * (d + (p-d)) * kappa_2 / (4 * c_k**2)

MSPSA_norm_theta_bound = np.concatenate(([norm_theta_0**2], MSPSA_norm_theta_bound))

### plot ###
linewidth = 2
plot_theta = plt.figure()
plt.grid()
plt.title(r'Mean Squared Error for $\mathbf{\theta}$')
# plt.title('Noralized Mean Squared Error for theta')
plt.ylabel("Mean Squared Error")
plt.plot(MSPSA_norm_theta, linewidth=linewidth, linestyle="-", color="black")
plt.plot(MSPSA_norm_theta_bound, linewidth=linewidth, linestyle="--", color="black")
plt.legend(["MSPSA", "Finite Sample Upper Bound"])
# plt.show()
plt.close()
plot_theta.savefig("figure/skewed_quartic_plot_MSPSA_bound_2020_08_07.pdf", bbox_inches='tight')






