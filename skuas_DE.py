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


def logistic_curve(t, t_shift, sigma, min_value, max_value): #TODO the max value here is actually min+max
    return (max_value/min_value)/(1+np.exp(-sigma*(t-t_shift))) + min_value

def next_function():
    pass

if __name__ == '__main__':
    y0 = [14,12]


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