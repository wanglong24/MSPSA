#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:06:06 2020

@author: Long Wang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 12})

skewed_quartic = np.load("data/skewed_quartic.npz")
theta_star = skewed_quartic["theta_star"]
loss_star = skewed_quartic["loss_star"]
theta_0 = skewed_quartic["theta_0"]
loss_0 = skewed_quartic["loss_0"]

MSPSA = np.load("data/skewed_quartic_MSPSA.npz")
MSPSA_norm_theta = MSPSA["norm_theta"]
MSPSA_norm_loss = MSPSA["norm_loss"]

RS = np.load("data/skewed_quartic_RS.npz")
RS_norm_theta = RS["norm_theta"]
RS_norm_loss = RS["norm_loss"]

SR = np.load("data/skewed_quartic_SR.npz")
SR_norm_theta = SR["norm_theta"]
SR_norm_loss = SR["norm_loss"]

iter_num = RS_norm_theta.shape[0]
# adjust for number of function measurements
x_axis_value = np.arange(0, iter_num+1)
MSPSA_norm_theta = np.concatenate(([1], np.repeat(MSPSA_norm_theta,2)))
MSPSA_norm_loss = np.concatenate(([1], np.repeat(MSPSA_norm_loss,2)))

RS_norm_theta = np.concatenate(([1], RS_norm_theta))
RS_norm_loss = np.concatenate(([1], RS_norm_loss))

M_ks = np.ceil(0.5 * np.log(np.arange(iter_num) + 2))
M_ks = M_ks.astype(np.int)

SR_norm_theta = np.concatenate(([1], np.repeat(SR_norm_theta, M_ks)))
SR_norm_theta = SR_norm_theta[:(iter_num+1)]
SR_norm_loss = np.concatenate(([1], np.repeat(SR_norm_loss, M_ks)))
SR_norm_loss = SR_norm_loss[:(iter_num+1)]

linewidth = 2
plot_theta = plt.figure()
plt.grid()
plt.title(r'Noralized Mean Squared Error for $\mathbf{\theta}$')
# plt.title('Noralized Mean Squared Error for theta')
plt.ylabel("Normalized Mean Squared Error")
plt.ylim(0, 1)
plt.plot(MSPSA_norm_theta, linewidth=linewidth, linestyle="-", color="black")
plt.plot(RS_norm_theta, linewidth=linewidth, linestyle="--", color="black")
plt.plot(SR_norm_theta, linewidth=linewidth, linestyle=":", color="black")
plt.legend(["MSPSA", "Random Search", "Stochastic Ruler"])
# plt.show()
plt.close()
plot_theta.savefig("figure/skewed_quartic_plot_theta_2020_08_06.pdf", bbox_inches='tight')

plot_loss = plt.figure()
plt.grid()
plt.title("Normalized Error for Loss")
plt.ylabel("Normalized Error")
plt.ylim(0, 1)
plt.plot(MSPSA_norm_loss, linewidth=linewidth, linestyle="-", color="black")
plt.plot(RS_norm_loss, linewidth=linewidth, linestyle="--", color="black")
plt.plot(SR_norm_loss, linewidth=linewidth, linestyle=":", color="black")
plt.legend(["MSPSA", "Random Search", "Stochastic Ruler"])
# plt.show()
plt.close()
plot_loss.savefig("figure/skewed_quartic_plot_loss_2020_08_06.pdf", bbox_inches='tight')