import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl

def generate_CI(data, function, f_kwargs, samples=5000, alpha=0.95):
    values = []
    for i in tqdm(range(samples)):
        sample = data.sample(data.shape[0], replace=True)
        value = function(sample, **f_kwargs)
        values.append(value)
    values = np.array(values)
    
    l = (1-0.95)/2
    u = 1-l
    return {'values':values, 'l': np.quantile(values, l), 'u':np.quantile(values, u)}


def plot_CI(values=None, l=0, u=1, plot=('m'), title='CI', xlim_l=None, xlim_u=None):
    if xlim_l is None: xlim_l = l
    if xlim_u is None: xlim_u = u
    xlim = (xlim_l, xlim_u)
    plt.plot((l, u), (1,1), 'co-')
    
    total = {'m':{'name': 'median',
                 'func': lambda x: np.median(x)},
            'a':{'name': 'average',
                 'func': lambda x: x.mean()}}
    
    anything = False
    for k,v in total.items():
        if k in plot:
                anything = True
                point = v['func'](values)
                label = v['name']
                plt.scatter(point, 1, label=label)
                
    plt.yticks([],[])
    plt.title(title)
    plt.xlim(xlim)
    if anything: plt.legend()
    plt.show()