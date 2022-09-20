# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 01:02:51 2021

@author: light
"""
from concurrent import futures
import matplotlib.pyplot as plt
import hyperspy.api as hs
from EELSFits import geteVrange
from STEMFactorize import PCA
import numpy as np
import h5py as hf
import os
from RL_deconvolution import RL_deconvl
from dask.distributed import Client
import pandas as pd
import dask.dataframe as dd



class importSTEM:
    def __init__ (self, name, path, dims = [0,0]):
        self.name = name
        self.path = path
        self.dims = dims
        
        if self.name.endswith('.dm3') or self.name.endswith('.dm4') or self.name.endswith('.dm5'):
            self.dataset, self.dims, self.cals, self.units = self.readDM()
        elif self.name.endswith('.hd5'):
            self.dataset, self.dims, self.cals, self.units = self.readhdf5()
        elif self.name.endswith('.csv'):
            self.dataset, self.dims, self.cals, self.units = self.readCSV()
        else:
            try:
                name.index('.')
            except ValueError:
                raise ValueError("File extension missing from name!")
            else:
                raise ValueError("Invalid file type!")
    
    def readDM(self):
        try:
            self.DMdata = hs.load(os.path.expanduser(self.path + '/'+ self.name))
        except (FileNotFoundError):
            print("File not Found. Make sure path and filename are correct.")
        
        self.cal = self.DMdata.axes_manager
        self.dims = self.DMdata.data.shape
        
        self.cals = np.array([np.array([self.cal[i].offset, self.cal[i].scale]) for i in range(len(self.dims))])
        self.units = [self.cal[i].units for i in range(len(self.dims))]
        
        return self.DMdata.data, self.dims, self.cals, self.units
    
    def readhdf5(self):
        with hf.File(self.path + "\\" + self.name + ".h5", 'r') as hdf:
            self.ls = list(hdf.keys())
            
            #print('List of datasets in this file: \n', ls)
            if len(self.ls) > 4:
                self.Y = np.array(hdf.get('Latent Variables'))
                self.M = np.array(hdf.get('Eigenvectors'))
                self.stdev = np.array(hdf.get('Standard Deviation'))
                self.mean = np.array(hdf.get('Mean'))
                
                self.Xr = np.multiply(np.dot(self.Y, self.M), self.stdev) + self.mean
                self.SI = self.Xr.reshape(self.dims)
            else:
                self.SI = np.array(hdf.get('SI'))
                
            self.dims = np.array(hdf.get('Dimensions'))
            self.cals = np.array(hdf.get('Calibrations'))
            self.asciiUnits = np.array(hdf.get('Units'))
            
        self.units = [ascii.decode('utf-8') for ascii in self.asciiUnits]
        
        return self.SI, self.dims, self.cals, self.units
    
    def exporthdf5(self):
        self.asciiUnits = [u.encode("ascii", "ignore") for u in self.units]

        with hf.File(self.path + "\\" + self.name[:-4] + ".h5", 'w') as hdf:
            print("Exporting Data to hdf5")
            hdf.create_dataset('SI', data = self.dataset)
            hdf.create_dataset('Dimensions', data = self.dims)
            hdf.create_dataset('Calibrations', data = self.cals)
            hdf.create_dataset('Units', data = self.asciiUnits)
    
    def exporthdf5_DM(self):
        f1 = hf.File(self.name + ".h5", "w")
        dset1 = f1.create_dataset("data", data = self.dataset)
        #dset1.attrs['scale'] = 0.01
        #dset1.attrs['offset'] = 15
        f1.close()
    
    def readCSV(self):
        try:
            self.data = np.genfromtxt(os.path.expanduser(self.path + '/'+ self.name), delimiter=',', skip_header = 1)
        except (FileNotFoundError):
            raise ValueError("File not Found. Make sure path and filename are correct.")
        
        self.xdata = self.data[:, 0].T
        self.dims = tuple(np.append(self.dims, len(self.xdata)))
        self.data3D = self.data[:, 1 :].T.reshape(self.dims[0], self.dims[1], self.dims[2])
        
        self.cals = np.array([[0, 0.5], [0, 0.5], [self.xdata[0], self.xdata[1] - self.xdata[0]]])
        self.units = ['nm', 'nm', 'meV']
        
        return self.data3D, self.dims, self.cals, self.units
    
    def checkPath(self, path2):
        self.path2 = path2
        
        if not os.path.exists(self.path2):
            os.makedirs(self.path2)
    
    def getData(self):
        return self.dataset
    
    def getDims(self):
        return self.dims
    
    def getCals(self):
        return self.cals
    
    def getUnits(self):
        return self.units

class importEELS(importSTEM):
    def __init__(self, name, path, dims):
        super().__init__(name, path, dims)
        self.norm_value = None
    
    def readDM(self):
        super().readDM()
        
        if self.units[2] == 'eV' and self.cals[2, 1] < 0.001:
            a = 1000
            self.units[2] = 'meV'
        else:
            a = 1
        
        self.cals[2] *= a
        
        return self.DMdata.data, self.dims, self.cals, self.units
    
    #correct Trim method is in DataSetFromSI
    def trim(self, croprange):
        if len(croprange) > 2:
            raise ValueError("Crop range should have format (x1, x2)")
            
        xdata = self.getEnergyAxis()
        ydatas = self.dataset
        
        I = geteVrange(croprange, xdata)
        #print(ydatas.shape)
        
        ytrim = ydatas[:, :, I[0] : I[1]]
        
        #print(self.dims[0])
        self.dims = (self.dims[0], self.dims[1], len(ytrim[0, 0, :]))
        self.dataset = ytrim
        self.cals[2, 0] = xdata[I[0]]
        
    def getEnergyAxis(self):
        return np.linspace(0, self.dims[2] - 1, self.dims[2]) * self.cals[2, 1] + self.cals[2, 0]


    ###########
    def RL(self, kerneltype=4, ZLP_data = None , FWHM=5, iterations=30, algo = 'hyperspy'):

        # if type(ZLP_data) != 'STEMhandler.importEELS':
        #     raise TypeError('Please input ZLP data as importEELS type.')

        client = Client()  # start local workers as processes

        
        if ZLP_data != None:
            ZLP = ZLP_data.dataset
            xk = ZLP_data.getEnergyAxis() # loading ZLP energy 
            if abs(xk[0]) < 1: #if units are in eV, make them meV
                xk *= 1000

            """roll kernel so that peak is centered"""
            kernelmax = np.where(k == max(k))[0][0]
            shift = int(k.size / 2) - kernelmax
            k = np.roll(k, shift)
            g = k  #use ZLP to deconvolve


        E_axis = self.getEnergyAxis()
        Raw_data = self.dataset

#         L = []
#         for row in Raw_data:
#             ll = []
#             for col in row:
#                 future = client.submit(RL_deconvl, E_axis, col, kerneltype=4, ZLP_path=ZLP_path , iterations=iterations,algo = algo)
#                 ll.append(future)
#             L.append(ll)

        if ZLP_data.dims[0:2] == (1,1):
            futures = [[client.submit(RL_deconvl, E_axis, col, kerneltype, ZLP_data=ZLP[0,0] , iterations=iterations,algo = algo) for col in row] for row in Raw_data]
        else:
            futures = [[client.submit(RL_deconvl, E_axis, col, kerneltype, ZLP_data=colZLP , iterations=iterations,algo = algo) for col,colZLP in zip(row,r2)] for row,r2 in zip(Raw_data,ZLP)]


        RL_all = np.array([[i.result() for i in future] for future in futures])
#         RL_all = np.array([[future.result() for future in ll] for ll in L] )    

        self.RLdata = RL_all
        client.close()
        return RL_all

    ###########
    def spectrum(self, E_axis, Rawdata3D, Deconvl_data, x_d, y_d, Range=None):
        if Range!=None:
            Range_index = geteVrange(Range,E_axis)
        else:
            Range_index = [0,-1]
        plt.plot( E_axis[Range_index[0]:Range_index[1]], Deconvl_data[x_d,y_d,Range_index[0]:Range_index[1]], color= 'b',label='Sub_RL')
        plt.plot( E_axis[Range_index[0]:Range_index[1]], Rawdata3D[x_d,y_d,Range_index[0]:Range_index[1]],color = 'r', label = 'Sub')
        plt.legend()
        plt.show()
        plt.clf()
       

    def PCAfilter(self, d):
        pca = PCA(d = d, data = self.dataND())
        smoothed = pca.smoothData()
        self.dataset = smoothed.reshape(self.dims[0], self.dims[1], self.dims[2])
    
    def dataND(self):
        return self.dataset.reshape(self.dims[0] * self.dims[1], self.dims[2])
    
    def exportCSV(self):
        self.xdata = self.getEnergyAxis()
        self.columns = self.dims[0] * self.dims[1]
        
        self.reshapeddata = self.dataset.reshape(self.columns, self.dims[2])
        
        self.savedata = np.vstack((self.xdata, self.reshapeddata)).T
        
        self.labels = np.array(['S' + str(i+1).zfill(4) for i in range(self.columns)])
        self.labels = np.append('Energy', self.labels)
        self.namesheader = ",".join(self.labels)
        
        np.savetxt(self.path + '\\' + self.name[:-4] + '_PCA.csv', self.savedata, delimiter = ',', header = self.namesheader)        
    
    def normalize(self, axis, sum=False):
        #self.dataset = np.divide(self.dataset, np.max(self.dataset, axis = axis))
        if sum:
            self.norm_value = np.sum(self.dataset, axis = axis)
        else:
            self.norm_value = np.max(self.dataset, axis = axis)
        norm_bc = np.einsum('kij->ijk', np.broadcast_to(self.norm_value, (self.dims[2], self.dims[0], self.dims[1])))
        self.dataset = np.divide(self.dataset, norm_bc)
        
        #self.dataset = np.array([self.dataset[:, :, i] /  np.max(self.dataset, axis = axis) for i in range(self.dims[2])])
    
    def unnormalize(self, axis, sum=False):
        #self.dataset = np.divide(self.dataset, np.max(self.dataset, axis = axis))
        if self.norm_value.all()==None:
            raise ValueError('You must first call normalize function before calling unnormalize on EELS object')
        else:
            norm_bc = np.einsum('kij->ijk', np.broadcast_to(self.norm_value, (self.dims[2], self.dims[0], self.dims[1])))
            self.dataset = np.multiply(self.dataset, norm_bc)
        self.norm_value=None
        

    def integrate(self, axis):        
        if axis == 0 or axis == 1:
            self.dataset = np.array([np.sum(self.dataset, axis = axis)])
            self.temp = list(self.dims)
            self.temp[axis] = 1
            self.dims = tuple(self.temp)
            print(self.dataset.shape)
        else:
            raise ValueError("Invalid axis!")
    
class import4DSTEM(importSTEM):
    def __init__(self, name, path):
        super().__init__(name, path)
        
    def readCSV(self):
        print(".csv is an invalid datatype for 4DSTEM data")
        
    def trim(self):
        pass
    
    def exportND(self):
        return self.dataset.reshape(self.dim[0] * self.dim[1], self.dim[2] * self.dim[3]).T
