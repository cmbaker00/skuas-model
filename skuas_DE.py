import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple


def model(y, t, params):
    r = y[0]
    pred1 = y[1]
    k = params.carrying_capacity_function(t, params)
    drdt = params.prey_growth_rate*r*(1-r/k) - .18*r*pred1
    dp1dt = params.prey_growth_rate*pred1*(1-pred1/(k*r))
    return [drdt, dp1dt]


def logistic_curve(t, t_shift, sigma, min_value, max_value):
    return (max_value-min_value)/(1+np.exp(-sigma*(t-t_shift))) + min_value


def carrying_capacity(t, t_control, sigma, min_capacity, max_capacity):
    return max_capacity if t < t_control else logistic_curve(t - t_control, np.log(99) / sigma, sigma, min_capacity, max_capacity)


def carrying_capacity_params(t, params):
    return carrying_capacity(t, params.erad_time, params.sigma,
                                          params.min_carrying_capacity,
                                          params.prey_carrying_capacity)

def plot_carrying_capacity(t_control, sigma, min_capacity, max_capacity, time_horison = 25):
    t_vals = np.arange(0, time_horison, 0.01)
    yvals = [carrying_capacity(t, t_control, sigma, min_capacity, max_capacity) for t in t_vals]
    plt.plot(t_vals, yvals)
    plt.xlabel('Years')
    plt.ylabel('Carrying capacity')
    plt.show()


if __name__ == '__main__':
    plot_carrying_capacity(t_control=2, sigma=.5, min_capacity=1, max_capacity=10)

    y0 = (15,10)
    param_struct = namedtuple('Parameters', ['prey_growth_rate', 'prey_carrying_capacity',
                                             'erad_time','min_carrying_capacity',
                                             'carrying_capacity_function','sigma'])
    params = param_struct(prey_growth_rate=1, prey_carrying_capacity=15,
                          erad_time=2, min_carrying_capacity=y0[0]/2,
                          carrying_capacity_function=carrying_capacity_params, sigma = .5)


    param = .1
    t = np.linspace(0,20,1000)
    y = odeint(lambda y, t: model(y,t,params), y0, t)

    # plot results
    plt.plot(t,y)
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.show()