"""
Created on Mon Nov 29 04:50:46 2021

@author: light
"""
import matplotlib as plt
from STEMhandler import importEELS
# from STEMFactorize import PCA

path = r"/Users/jiatizhao/Desktop/Pan's_Lab/Chait's library"
name = '01_phonon_1s_3DW_Spectrum Image (EELS) (aligned).dm4'
plots = [1, 10, 50]

VibData = importEELS(name, path)

# print(VibData.RL(kerneltype=4, ZLP_path=ZLP,iteration=50)
print(VibData.getDims())
print(VibData.getCals())
print(VibData.getUnits())
plt.plot(VibData.getEnergyAxis(), VibData.dataset[10, 51])

#(y, x, E)
VibData.integrate(axis = 0)
VibData.normalize(axis = 2)
VibData.trim((5, 150))

print(VibData.getDims())
print(VibData.getCals())
print(VibData.getUnits())

VibData.PCAfilter(d = 10)

VibData.exportCSV()

'''PCA Analysis'''
'''
dataND = VibData.dataND()
xdata = VibData.getEnergyAxis()
dims = VibData.getDims()

pca = PCA(d = 10, data = dataND) #create PCA object

pca.plotAll(plots, dims, xdata, name, path, scaled = True, exportCSV = False)

print(pca.eigenValues()) #return matrix containing eigenvalues as columns
print(pca.eigenVectors()) #return matrix containing eigenvectors as rows
print(pca.eigenMap(n = 1, dims = dims)) #map of eigenvalues belonging to eigenvector n
print(pca.smoothData())

pca.vectorPlots(xdata, name, path, scaled = True, exportCSV = True) #stacked eigenvector plots
pca.scree(name, path, exportCSV = True) #scree and cumulative variance
pca.pairPlot(name, path) #Pair plots (for viewing d-dimensional diagonalized data from different 2D projections)
pca.compare(plots, xdata, name, path, exportCSV = True) #choose which data point to compare
pca.eigenMapPlots(dims, path, name, exportCSV = True)
'''
