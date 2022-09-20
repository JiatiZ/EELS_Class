# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:20:56 2021

@author: light
"""
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np




def geteVrange(ranges, xdata):
    rangesX = [(np.abs(np.array(xdata) - ranges[s])).argmin() for s in range(len(ranges))]
    
    return rangesX
    
def fitRegions(ranges, xdata, ydata):
    X = geteVrange(ranges, xdata)
    
    xcut = []
    ycut = []
    for i in range(int(len(X)/2)):
        xcut = np.append(xcut, xdata[X[2*i] : X[2*i+1]])
        ycut = np.append(ycut, ydata[X[2*i] : X[2*i+1]])
        
    return xcut, ycut

def powerFit(xdata, ydata, ranges):
    def powerLaw(x, A, B):
        return A*x**B
    
    xcut, ycut = fitRegions(ranges, xdata, ydata)
    xcurve, ycurve = fitRegions([ranges[0], xdata[-1]], xdata, ydata)
    
    try:
        params, covariance = curve_fit(powerLaw, xcut, ycut, maxfev = 10000)
    except (RuntimeError, ValueError) as e:
        params = errors = [0, 0]
        print("Fitting Failed: " + str(e))
    else:
        errors = np.sqrt(np.diag(covariance))
        
    print(params)
    print(errors)
    
    bksubbed = ycurve - powerLaw(xcurve, *params)
    
    plt.plot(xdata, ydata)
    plt.plot(xcut, ycut, 'o', markersize = 4)
    plt.plot(xcurve, powerLaw(xcurve, *params))
    plt.plot(xcurve, bksubbed)
    plt.xlim([35, 80])
    plt.ylim([-0.0001, 0.004])
    
    return bksubbed

def powerFitAll(xdata, ydatas, ranges):
    import dask
    from distributed import Client
    
    client = Client(n_workers = 12, threads_per_worker = 1)
    client
    
    bksubbedAll = [[dask.delayed(powerFit)(xdata, ydata, ranges) for ydata in rows] for rows in ydatas]
    bksubbedAllCompute = dask.compute(*bksubbedAll)
    
    client.close()
    
    return np.array(bksubbedAllCompute)

def rootFit(xdata, ydata, ranges):
    def rootAndLine(xv, n, A, B, C):
        return [A * (x-B)**n + C if x > B else C for x in xv]
    
    xcut, ycut = fitRegions(ranges, xdata, ydata)
    xcurve, ycurve = fitRegions([ranges[0], xdata[-1]], xdata, ydata)
    
    #p0 = [0.5, max(ycut), 2, min(ycut)]
    p0 = [0.5, 0.001, 2, 0.001]
    p0 = [7.16786789e-01, 7.00794561e-04, 2.79000034e+00, 1.12557920e-03]
    
    try:
        params, covariance = curve_fit(rootAndLine, xcut, ycut, p0 = p0, maxfev = 10000)
    except (RuntimeError, ValueError) as e:
        params = errors = [0, 0, 0, 0]
        print("Fitting Failed: " + str(e))
    else:
        errors = np.sqrt(np.diag(covariance))
    
    bckgrnd = ycurve - rootAndLine(xcurve, *params)
    
    plt.plot(xdata, ydata)
    plt.plot(xcut, ycut, 'o', markersize = 4)
    plt.plot(xcurve, rootAndLine(xcurve, *params))
    plt.plot(xcurve, bckgrnd)
    plt.xlim([xdata[0], xdata[-1]])
    plt.ylim([-0.001, 0.015])
    
    return params

def rootFitAll(xdata, ydatas, ranges):
    import dask
    from distributed import Client
    
    client = Client(n_workers = 32, threads_per_worker = 1)
    client
    
    paramsAll = [[dask.delayed(rootFit)(xdata, ydata, ranges) for ydata in rows] for rows in ydatas]
    paramsAllCompute = dask.compute(*paramsAll)
    
    client.close()
    
    return np.array(paramsAllCompute)

def exportCSV(data, xdata, path, name):
    dims = data.shape
    columns = dims[0] * dims[1]
    
    reshapeddata = data.reshape(columns, dims[2])
    
    savedata = np.vstack((xdata, reshapeddata)).T
    
    labels = np.array(['S' + str(i+1).zfill(4) for i in range(columns)])
    labels = np.append('Energy', labels)
    namesheader = ",".join(labels)
    
    np.savetxt(path + '\\' + name[:-4] + '.csv', savedata, delimiter = ',', header = namesheader)