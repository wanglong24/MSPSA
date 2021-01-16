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

str_today = str(np.datetime64('today'))

skewed_quartic = np.load("data/skewed-quartic.npz")
theta_star = skewed_quartic["theta_star"]
loss_star = skewed_quartic["loss_star"]
theta_0 = skewed_quartic["theta_0"]
loss_0 = skewed_quartic["loss_0"]

MSPSA = np.load("data/skewed-quartic-MSPSA.npz")
MSPSA_theta_norm_error = MSPSA["theta_norm_error"]
MSPSA_loss_norm_error = MSPSA["loss_norm_error"]

RS = np.load("data/skewed-quartic-RS.npz")
RS_theta_norm_error = RS["theta_norm_error"]
RS_loss_norm_error = RS["loss_norm_error"]

SR = np.load("data/skewed-quartic-SR.npz")
SR_theta_norm_error = SR["theta_norm_error"]
SR_loss_norm_error = SR["loss_norm_error"]

meas_num = RS_theta_norm_error.shape[0]
x_axis_value = np.arange(0, meas_num+1)

# adjust for number of function measurements
MSPSA_theta_norm_error = np.concatenate(([1], np.repeat(MSPSA_theta_norm_error,2)))
MSPSA_loss_norm_error = np.concatenate(([1], np.repeat(MSPSA_loss_norm_error,2)))

RS_theta_norm_error = np.concatenate(([1], RS_theta_norm_error))
RS_loss_norm_error = np.concatenate(([1], RS_loss_norm_error))

M_ks = np.load("data/skewed-quartic-SR-iter-seq.npy")
SR_theta_norm_error = np.concatenate(([1], np.repeat(SR_theta_norm_error, M_ks)))
SR_theta_norm_error = SR_theta_norm_error[:(meas_num+1)]
SR_loss_norm_error = np.concatenate(([1], np.repeat(SR_loss_norm_error, M_ks)))
SR_loss_norm_error = SR_loss_norm_error[:(meas_num+1)]

# plot
linewidth = 2
plot_theta = plt.figure()
plt.grid()
plt.title(r'Normalized Mean-Squared Error for $\hat{\mathbf{\theta}}_k$')
plt.xlabel("Number of Loss Function Measurements")
plt.ylabel("Normalized Mean-Squared Error")
plt.ylim(0, 1)
plt.plot(RS_theta_norm_error**2, linewidth=linewidth, linestyle=":", color="black")
plt.plot(SR_theta_norm_error**2, linewidth=linewidth, linestyle="-", color="black")
plt.plot(MSPSA_theta_norm_error**2, linewidth=linewidth, linestyle="--", color="black")
plt.legend(["Local Random Search", "Stochastic Ruler", "MSPSA"])
plt.close()
plot_theta.savefig("figure/skewed-quartic-theta-error-"+str_today+".pdf", bbox_inches='tight')

plot_loss = plt.figure()
plt.grid()
plt.title("Normalized Error for Loss")
plt.xlabel("Number of Loss Function Measurements")
plt.ylabel("Normalized Error")
plt.ylim(0, 1)
plt.plot(RS_loss_norm_error, linewidth=linewidth, linestyle=":", color="black")
plt.plot(SR_loss_norm_error, linewidth=linewidth, linestyle="-", color="black")
plt.plot(MSPSA_loss_norm_error, linewidth=linewidth, linestyle="--", color="black")
plt.legend(["Local Random Search", "Stochastic Ruler", "MSPSA"])
plt.close()
plot_loss.savefig("figure/skewed-quartic-loss-error-"+str_today+".pdf", bbox_inches='tight')

print("Done printing.")