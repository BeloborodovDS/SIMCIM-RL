import numpy as np
import torch

def tanh_pump(O,S,D,Jmax,amin=0,amax=-1,N=1000):
    i = torch.arange(N,dtype=torch.float32)
    arg = torch.tensor(S,dtype = torch.float32)*(i/N-0.5)
    ret = (Jmax*O*(torch.tanh(arg).numpy() + D) + amin)/(amin-amax)
    return lambda i: float(ret[i])

def read_gset(filename, negate=True):
    """Read Gset graph from file"""
    with open(filename) as f:
        data = f.read().strip('\n').split('\n')
    data = [e.strip().split(' ') for e in data]
    n, nonzero = data[0]
    n, nonzero = int(n), int(nonzero)
    data = data[1:]
    ii, jj, vals = zip(*data)
    ii = [int(i) for i in ii]
    jj = [int(i) for i in jj]
    vals = [float(e) for e in vals]
    A = np.zeros((n,n), dtype=float)
    A[np.array(ii)-1, np.array(jj)-1] = np.array(vals)
    A[np.array(jj)-1, np.array(ii)-1] = np.array(vals)
    if negate:
        A = -A
    return A

def read_gbench(fn):
    """read BLS benchmarks from file"""
    with open(fn) as f:
        names,values = f.read().split('\n\n')
    names = names.split('\n')
    values = values.split('\n')
    return {int(n.strip('G')):int(v) for n,v in zip(names,values)}

