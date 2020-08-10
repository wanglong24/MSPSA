#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 02:03:13 2020

@author: longwang
"""

import numpy as np

class BusSchedule:
    def __init__(self, rate):
        self.rate = rate # passanger arrival time follows poisson(10)

    def loss(self, theta):
        passanger_time = []
        current_time = 0
        while current_time < 100:
            current_time += np.random.exponential(1/self.rate)
            passanger_time.append(current_time)

        bus_time = np.sort(theta)
        bus_time = np.concatenate((bus_time[:4],[50],bus_time[4:],[100]))
        total_wait_time = 0
        for i in range(len(passanger_time)):
            bus_arrival_time = bus_time[np.argmax(bus_time > passanger_time[i])]
            total_wait_time += bus_arrival_time - passanger_time[i]

        return total_wait_time

if __name__ == "__main__":
    rate = 10
    bus_schedule = BusSchedule(rate)

    theta = np.array([10,20,30,40,175/3,200/3,225/3,250/3,275/3]) # loss = 4488-4494
    theta = np.array([10,15,20,35,175/3,200/3,225/3,250/3,275/3]) # loss = 5000

    true_loss = 0
    n = 1000
    for i in range(n):
        true_loss += bus_schedule.loss(theta) / n
    print(true_loss)
