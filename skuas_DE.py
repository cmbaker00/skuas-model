import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple


def model(y, t, params):
    r = y[0]
    pred1 = y[1]
    k = params.preycarryingcapacity if (t < params.erad_time) else params.min_carrying_capacity
    drdt1 = params.preygrowthrate*r*(1-r/k) - .18*r*pred1
    drdt2 = params.preygrowthrate*pred1*(1-pred1/(k*r))
    return [drdt1, drdt2]


def logistic_curve(t, t_shift, sigma, min_value, max_value):
    return (max_value-min_value)/(1+np.exp(-sigma*(t-t_shift))) + min_value


def carrying_capacity(t, t_control, sigma, min_capacity, max_capacity):
    return max_capacity if t < t_control else logistic_curve(t - t_control, np.log(99) / sigma, sigma, min_capacity, max_capacity)



def plot_carrying_capacity(t_control, sigma, min_capacity, max_capacity, time_horison = 25):
    t_vals = np.arange(0, time_horison, 0.01)
    yvals = [carrying_capacity(t, t_control, sigma, min_capacity, max_capacity) for t in t_vals]
    plt.plot(t_vals, yvals)
    plt.show()


if __name__ == '__main__':
    plot_carrying_capacity(t_control=2, sigma=5, min_capacity=1, max_capacity=10)


    param_struct = namedtuple('Parameters', ['preygrowthrate', 'preycarryingcapacity',
                                             'erad_time','min_carrying_capacity'])
    params = param_struct(preygrowthrate=1, preycarryingcapacity=15, erad_time=2, min_carrying_capacity=y0[0]/2)


    param = .1
    t = np.linspace(0,20,1000)
    y = odeint(lambda y, t: model(y,t,params), y0, t)

    # plot results
    plt.plot(t,y)
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.show()