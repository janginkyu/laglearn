#!/home/janginkyu/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sin, cos, sqrt, tanh

def run_simulation(plot=False):

    m = 13.0
    k = 16.0

    def u(t):
        return sin(t) + 2.0 * cos(sqrt(2.0) * t) - cos(4.32 * t - 3.2)

    def friction(x):
        return 2.0 * x[1] + 3.0 * tanh(x[1] * 3.0)

    def accel(tx):
        return -(k * tx[1] + friction(tx[1:3]) - u(tx[0])) / m

    def force(tx):
        return u(tx[0])

    # pos:x[0] / vel:x[1]
    def dx_dt(x, t):
        dxdt = np.array([0.0, 0.0])
        dxdt[0] = x[1]
        dxdt[1] = accel(np.array([t, x[0], x[1]]))
        return dxdt

    ts0 = np.linspace(0.0, 150.0, 10000)
    x0 = np.array([1.0, 0.0])
    xs = odeint(dx_dt, x0, ts0)
    xs = np.array(xs)
    ts = np.array([np.array(ts0)])
    tx = np.concatenate((ts.T, xs), axis=1)
    accl = np.array([np.apply_along_axis(accel, 1, tx)])
    us = np.array([np.apply_along_axis(force, 1, tx)])
    
    data = np.concatenate((xs, accl.T, us.T, ts.T), axis=1)

    if plot:
        plt.rcParams.update({'font.size': 14})
        plt.xlabel("t")
        plt.ylabel("x")
        plt.plot(ts0, xs)
        plt.plot(ts[0], accl)
        plt.plot(ts[0], us)
        plt.show()

    return data

dat = run_simulation()
print(dat[:, 0].size)
