# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 02:32:00 2021

@author: light
"""
from sklearn import decomposition
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import h5py as hf
import numpy as np
import os


class factorize:
    def __init__(self, d, data):
        self.d = d
        self.data = data
        
        self.mu = np.mean(self.data, axis = 0)
        self.stdev = np.std(self.data, axis = 0)
        
    def checkPath(self, path2):
        self.path2 = path2
        
        if not os.path.exists(self.path2):
            os.makedirs(self.path2)
    
    def scale(self):
        self.stdev[self.stdev == 0] = 1
        
        return np.divide(self.data - self.mu, self.stdev)
    
    def descale(self, scaleddata2):
        self.scaleddata2 = scaleddata2
        
        return (self.scaleddata2 * self.stdev) + self.mu
    
    def exporthdf5(self, name, path, dims, cals, units):
        self.dataset = self.smoothData()
        self.asciiUnits = [u.encode("ascii", "ignore") for u in self.units]

        with hf.File(path + "\\" + name + ".h5", 'w') as hdf:
            print("Exporting Data to hdf5")
            hdf.create_dataset('SI', data = self.dataset)
            hdf.create_dataset('Dimensions', data = self.dims)
            hdf.create_dataset('Calibrations', data = self.cals)
            hdf.create_dataset('Units', data = self.asciiUnits)
    
class PCA(factorize):
    def __init__(self, d, data):
        super().__init__(d, data)
        self.scaleddata = factorize.scale(self)
        self.fit = decomposition.PCA(n_components = self.d).fit(self.scaleddata)
        
    def eigenVectors(self):
        self.M = self.fit.components_
        return self.M
    
    def eigenValues(self):
        self.Y = self.fit.transform(self.scaleddata)
        return self.Y
    
    def setEigenVectors(self, newVecs):
        self.M = newVecs
        
    def setEigenValues(self, newVals):
        self.Y = newVals
        
    def eigenMap(self, n, dims):
        self.n = n
        self.dims = dims
        
        return self.eigenValues()[:, n].reshape((self.dims[0], self.dims[1]))
    
    def eigenMapPlots(self, dims, path, name, exportCSV = False):
        self.dims = dims
        self.name = name
        self.path = path
        self.exportCSV = exportCSV
        factorize.checkPath(self, self.path + '\\EigenMaps')
        
        for self.n in range(self.d):
            self.eMap = self.eigenMap(self.n, self.dims)
            
            self.ax = plt.gca()
            self.img = self.ax.imshow(self.eMap, origin = 'upper', extent = [0, self.dims[1], 0, self.dims[0]], interpolation = 'nearest', cmap = 'gray')
            #axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left', bbox_to_anchor = (1.01, 0., 1, 1), bbox_transform = ax.transAxes, borderpad = 0)
            #plt.colorbar(img, cax = axins)
            self.ax.axis('off')
            plt.savefig(self.path + "\\EigenMaps\\" + self.name[:-4] + '_PC' + str(self.n).zfill(2) + '.png', bbox_inches = 'tight', pad_inches = 0, dpi = 300)
            plt.clf()
        
            if self.exportCSV:
                np.savetxt(self.path + '\\EigenMaps\\' + self.name[:-4] + '_PC' + str(self.n).zfill(2) + '.csv', self.eMap, delimiter = ',')
    
    def vectorPlots(self, xdata, name, path, scaled = True, exportCSV = False):
        self.xdata = xdata
        self.name = name
        self.path = path
        self.scaled = scaled
        self.exportCSV = exportCSV
        
        factorize.checkPath(self, self.path)
        factorize.checkPath(self, self.path + '\\CSV Exports')
        
        if scaled:
            self.M = self.eigenVectors()
        else:
            self.M = factorize.descale(self, self.eigenVectors())
        
        for i in range(self.d):
            plt.subplot(self.d, 1, i + 1)
            plt.plot(self.xdata, self.M[i])                
            plt.grid(b = None)
      
        plt.savefig(self.path + '\\' + self.name[:-4] + '_vectorPlots.png', dpi = 500)
        plt.clf()
        
        if self.exportCSV:
            self.vectorCSV = np.vstack((self.xdata, self.M)).T
            np.savetxt(self.path + '\\CSV Exports\\' + self.name[:-4] + '_vectorPlots.csv', self.vectorCSV, delimiter = ',')
    
    def smoothData(self):
        self.M = self.eigenVectors()
        self.Y = self.eigenValues()
        
        return factorize.descale(self, np.dot(self.Y, self.M))
    '''
    def smoothData(self, dims):
        self.dims = dims
        self.Y = self.eigenVectors()
        self.M = self.eigenValues()
        
        return self.descale(np.dot(self.Y, self.M)).reshape(dims)
    '''
    def scree(self, name, path, exportCSV = False):
        self.name = name
        self.path = path
        self.exportCSV = exportCSV
        
        factorize.checkPath(self, self.path)
        factorize.checkPath(self, self.path + '\\CSV Exports')
        
        self.plotx = np.linspace(1, self.d, self.d)
        self.plotvar = self.fit.explained_variance_ratio_
        self.plotcum = np.cumsum(self.plotvar)
        
        """percentage of explained variance plots (scree plots)"""
        plt.bar(self.plotx, self.plotvar*100)
        plt.plot(self.plotx, self.plotvar*100, "go-")
        for i in range(self.d):
            plt.text(self.plotx[i], self.plotvar[i]*100 + 0.02, "%.1f" % round(self.plotvar[i] * 100, 2) + "%")
        plt.xlabel('number of components')
        plt.ylabel('Percentage of explained variance')
        plt.grid(b = None)
        plt.savefig(self.path + "\\" + self.name[:-4] + '_EV.png', dpi = 500)
        plt.clf()
        
        """cumulitive explained variance plots"""
        plt.plot(self.plotx, self.plotcum, "go-")
        for i in range(self.d):
            plt.text(self.plotx[i], self.plotcum[i], "%.1f" % round(self.plotcum[i] * 100, 2) + "%")
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.grid(b = None)
        plt.savefig(self.path + "\\" + self.name[:-4] + '_cumEV.png', dpi = 500)
        plt.clf()
        
        if self.exportCSV:
            self.cumvar = np.vstack((self.plotx, self.plotvar, self.plotcum)).T
            namesheader = "PC, Explained Variance, Cumulative Explained Variance"
            np.savetxt(self.path + '\\CSV Exports\\' + self.name[:-4] + '_Variance.csv', self.cumvar, delimiter = ',', header = namesheader)

    def pairPlot(self, name, path):
        self.name = name
        self.path = path
        factorize.checkPath(self, self.path)
        
        self.Y = self.eigenValues()
        
        # Plot pairs of latent vars.
        self.columns = ['y{}'.format(i) for i in range(self.d)]
        sns.pairplot(pd.DataFrame(self.Y, columns = self.columns))
        plt.grid(b = None)
        plt.savefig(self.path + '\\' + self.name[:-4] + '_dim.png', dpi = 200)
        plt.clf()
    
    def compare(self, plots, xdata, name, path, exportCSV = False):
        self.plots = plots
        self.xdata = xdata
        self.name = name
        self.path = path
        self.exportCSV = exportCSV
        
        #check if plots #s are within data
        
        factorize.checkPath(self, self.path)
        if exportCSV: factorize.checkPath(self, self.path + '\\CSV Exports')
        
        self.sdata = self.smoothData()
        
        """Compare Raw data with smoothed data."""
        # Compare a few samples from X and Xr.
        plt.figure(figsize = (8.5, 4))
        
        #for i,c in zip((75, 2550, 2521), sns.color_palette()):
        for i,c in zip(self.plots, sns.color_palette()):
            
            plt.xlabel('Feature #')
            plt.ylabel('Feature Value')
            label = '{}(d = {}): $\sigma = {:.4f}$'.format('PCA', self.d, np.std(self.sdata - self.data))
            plt.text(0.95, 0.9, label, horizontalalignment = 'right', fontsize = 'x-large', transform = plt.gca().transAxes)
                
            plt.plot(self.xdata, self.data[i], '-', c = c, ms = 3, alpha = 0.5)
            plt.plot(self.xdata, self.sdata[i], '-', c = c, lw = 2)
            plt.grid(b = None)
                
            #plt.xlim([10, 120])
            #plt.ylim([0.0001, 0.04])
            plt.savefig(self.path + '\\' + self.name[:-4] + '_' + str(i).zfill(2) + '.png', dpi = 200)
            plt.clf()
            
            if self.exportCSV:
                self.comparedPlots = np.vstack((self.xdata, self.data[i], self.sdata[i])).T
                np.savetxt(self.path + '\\CSV Exports\\' + self.name[:-4] + '_PC' + str(i).zfill(2) + '.csv', self.comparedPlots, delimiter = ',')
                
    def plotAll(self, plots, dims, xdata, name, path, scaled = True, exportCSV = False):
        self.plots = plots
        self.dims = dims
        self.xdata = xdata
        self.name = name
        self.path = path
        self.scaled = scaled
        self.exportCSV = exportCSV
        
        factorize.checkPath(self, self.path)
        if exportCSV: factorize.checkPath(self, self.path + '\\CSV Exports')
        
        self.vectorPlots(self.xdata, self.name, self.path, self.scaled, self.exportCSV) #stacked eigenvector plots
        self.scree(self.name, self.path, self.exportCSV) #scree and cumulative variance
        self.pairPlot(self.name, self.path) #Pair plots (for viewing d-dimensional diagonalized data from different 2D projections)
        self.compare(self.plots, self.xdata, self.name, self.path, self.exportCSV) #choose which data point to compare
        self.eigenMapPlots(self.dims, self.path, self.name, self.exportCSV)