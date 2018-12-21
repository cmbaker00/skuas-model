import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple


def model(y, t, params):
    # Define some parameters to make equations easier to read

    #State variables
    resource = y[0]
    pred1 = y[1]
    pred2 = y[2]
    pred3 = y[3]

    #Composite parameters
    k = params.carrying_capacity_function(t, params)
    death_rate = 1/params.lifespan
    food_availability = 1-(pred1+pred2+pred3)/resource

    # Model
    drdt = params.prey_growth_rate*resource*(1-resource/k) - \
           params.p1_consumption*pred1 - \
           params.p2p3_consumption*(pred2+pred3)
    dp1dt = params.p2_reprod_rate*pred2 + \
            params.p3_reprod_rate*pred3 - \
            params.mature_p1p2*pred1 - \
            death_rate/food_availability
    dp2dt = params.mature_p1p2*pred1 - \
            params.mature_p2p3*pred2 - \
            death_rate/food_availability
    dp3dt = params.mature_p2p3*pred2 - \
            death_rate/food_availability
    return [drdt, dp1dt,dp2dt,dp3dt]


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


def plot_carrying_capacity_params(params, time_horison = 25):
    plot_carrying_capacity(params.erad_time, params.sigma, params.min_carrying_capacity,
                           params.prey_carrying_capacity, time_horison)

if __name__ == '__main__':

    y0 = (15,1,2,2)
    param_struct = namedtuple('Parameters', ['prey_growth_rate', 'prey_carrying_capacity',
                                             'erad_time','min_carrying_capacity',
                                             'carrying_capacity_function','sigma',
                                             'mature_p1p2','mature_p2p3',
                                             'p1_consumption','p2p3_consumption','p2_reprod_rate',
                                             'p3_reprod_rate','lifespan'])
    params = param_struct(prey_growth_rate=1, prey_carrying_capacity=15,
                          erad_time=2, min_carrying_capacity=y0[0]*.4,
                          carrying_capacity_function=carrying_capacity_params, sigma = .11, mature_p1p2=1,
                          mature_p2p3=1/7,
                          p1_consumption=.15, p2p3_consumption=.2, p2_reprod_rate=0,
                          p3_reprod_rate=2,lifespan=30)

    plot_carrying_capacity_params(params)

    param = .1
    t = np.linspace(0,20,1000)
    y = odeint(lambda y, t: model(y,t,params), y0, t)

    # plot results
    plt.plot(t,y)
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.show()
