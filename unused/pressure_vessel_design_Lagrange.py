#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:11:27 2020

@author: longwang
"""

import numpy as np

class PressureVesselDesignLagrange:
    def __init__(self, multiplier=1.8, r=[1e-4,1e-4,1e-4,1e-4], lamb=0):
        self.multiplier = multiplier
        self.r_1 = r[0]
        self.r_2 = r[1]
        self.r_3 = r[2]
        self.r_4 = r[3]
        self.lamb_1 = lamb
        self.lamb_2 = lamb
        self.lamb_3 = lamb
        self.lamb_4 = lamb

    def loss(self, theta):
        x_1 = theta[0] * 0.0625
        x_2 = theta[1] * 0.0625
        x_3 = theta[2]
        x_4 = theta[3]

        L = 0.6224 * x_1 * x_3 * x_4 + 1.7781 * x_2 * x_3**2 \
            + 3.1661 * x_1**2 * x_4 + 19.84 * x_1**2 * x_3
        return float(L)

    def get_g(self, theta):
        x_1 = theta[0] * 0.0625
        x_2 = theta[1] * 0.0625
        x_3 = theta[2]
        x_4 = theta[3]

        g_1 = -x_1 + 0.0193 * x_3
        g_2 = -x_2 + 0.00954 * x_3
        g_3 = (-np.pi * x_3**2 * x_4 - 4/3 * np.pi * x_3**3 + 1296000) / 1296000
        g_4 = x_4 - 240

        return g_1, g_2, g_3, g_4

    def update_Lagrangian(self, theta):
        self.r_1 = self.multiplier * self.r_1
        self.r_2 = self.multiplier * self.r_2
        self.r_3 = self.multiplier * self.r_3
        self.r_4 = self.multiplier * self.r_4

        g_1, g_2, g_3, g_4 = self.get_g(theta)

        self.lamb_1 = self.lamb_1 + 2 * self.r_1 * max(g_1, -self.lamb_1 / (2*self.r_1))
        self.lamb_2 = self.lamb_2 + 2 * self.r_2 * max(g_2, -self.lamb_2 / (2*self.r_2))
        self.lamb_3 = self.lamb_3 + 2 * self.r_3 * max(g_3, -self.lamb_3 / (2*self.r_3))
        self.lamb_4 = self.lamb_4 + 2 * self.r_4 * max(g_4, -self.lamb_4 / (2*self.r_4))

    def loss_Lagrangian(self, theta):
        g_1, g_2, g_3, g_4 = self.get_g(theta)

        phi_1 = max(g_1, -self.lamb_1 / (2*self.r_1))
        phi_2 = max(g_2, -self.lamb_2 / (2*self.r_2))
        phi_3 = max(g_3, -self.lamb_3 / (2*self.r_3))
        phi_4 = max(g_4, -self.lamb_4 / (2*self.r_4))

        loss_Lagrangian_value = self.loss(theta)
        loss_Lagrangian_value += self.lamb_1 * phi_1 + self.r_1 * phi_1**2
        loss_Lagrangian_value += self.lamb_2 * phi_2 + self.r_2 * phi_2**2
        loss_Lagrangian_value += self.lamb_3 * phi_3 + self.r_3 * phi_3**2
        loss_Lagrangian_value += self.lamb_4 * phi_4 + self.r_4 * phi_4**2

        return loss_Lagrangian_value

    def show_g(self, theta):
        g_1, g_2, g_3, g_4 = self.get_g(theta)
        print("g_value:", g_1, g_2, g_3, g_4)

if __name__ == "__main__":
    # 0.8125 / 0.0625 = 13
    # 1.1250 / 0.0625 = 18
    # 0.4375 / 0.0625 = 7

    pressure_vessel_design = PressureVesselDesignLagrange()
    # theta = [1.125/0.0625, 0.625/0.0625, 58.291, 43.690]
    # theta = [1.125/0.0625, 0.625/0.0625, 47.700, 117.701]

    # WOA: 6059.7410
    theta = np.array([13, 7, 42.0984455958549, 176.6365958424394])

    # theta = np.array([18, 10, 50, 150])

    theta = [5.,6.,47.54745887, 149.74276268]

    print(pressure_vessel_design.loss(theta))
    print(pressure_vessel_design.loss_Lagrangian(theta))
    pressure_vessel_design.show_g(theta)
