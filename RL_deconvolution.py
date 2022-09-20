import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
import matplotlib.pyplot as plt
from skimage import color, data, restoration
'''
ZLP_data for Vacuum ZLP if the user has one
kerneltype options as follow:
    #1: Gaussian
    #2: Lorentzian
    #3: Voight
    #4: ZLP
FWHM of Gaussian or lorentzian to be used for deconvolution.
Iteration of Richardson-Lucy deconvolution to operate'''
def gaus(x, A, mu, sigma):
    return A* np.exp(-1/2 * ((x - mu) / sigma) ** 2)

def lor(x, A, mu, Gamma):
    return (A * (Gamma/2)**2 )/ ((x - mu)**2 + ((1/2) * Gamma)**2)

def pvoigt(x, A, mu, Gamma, n):
    sigma = Gamma / (2 * np.sqrt(2 * np.log(2)))
    return (n * gaus(x, A, mu, sigma) + (1 - n) * lor(x, A, mu, Gamma))

def RL_deconvl(x, y, kerneltype=1, ZLP_data = None , FWHM=5, iterations=30, algo = 'hyperspy'):
    dispf = x[1] - x[0] #dispersion of signal
    xg = np.linspace(-(x.size/2) * dispf, (x.size / 2) * dispf, x.size)
    def deconv_function(signal, kernel=None, iterations=15, psf_size=None):
            imax = kernel.argmax()
            result = np.array(signal).copy()
            mimax = psf_size - 1 - imax
            for _ in range(iterations):
                first = np.convolve(kernel, result)[imax: imax + psf_size]
                result *= np.convolve(kernel[::-1], signal /
                                      first)[mimax:mimax + psf_size]
            return result

    """choose kernel"""
    if ZLP_data.all() != None:
        """load vacuum ZLP as kernel"""
        k = ZLP_data
        # xk, k = np.genfromtxt(ZLP_data + ".csv", delimiter=',').T

        # if abs(xk[0]) < 1: #if units are in eV, make them meV
        #     xk *= 1000

        # dispk = xk[1] - xk[0] # dispersion of kernel
        # factor = dispf / dispk # disperison correction factor if different dispersions are used
        
        """build kernel"""
        
        if kerneltype == 1:
            p0 = [1, 0, 7]
            popt_gaus, pcov = curve_fit(gaus, xk, k, p0 = p0, maxfev=1000000)
            g = gaus(xg, *popt_gaus)
        elif kerneltype == 2:
            p0 = [1, 0, 7]
            popt, pcov = curve_fit(lor, xk, k, p0 = p0, maxfev=1000000)
            g = lor(xg, *popt)
        elif kerneltype == 3:
            p0 = [1, 0, 7, 0.5]
            popt, pcov = curve_fit(pvoigt, xk, k, p0 = p0, maxfev=1000000)
            g = pvoigt(xg, *popt)
        elif kerneltype == 4:
            """make data positive"""
            sigmin = min(y)
            y -= sigmin
            k -= min(k)    
            
            # """normalize areas of signal and deconvolution kernel"""
            # sigsum = sum(y)
            # y /= sigsum
            # k /= sum(k)
            
            sizediff = k.size - y.size
            if sizediff > 0:
                # cut the kernel so it has the same size as the signal
                kernelmax = np.where(k == max(k))[0][0]
                shift = int(k.size / 2) - kernelmax
                k = np.roll(k, shift)
                k = k[ : -sizediff]
                # k  = k[y.size//2 - sizediff : y.size//2 + sizediff]
                
            else:
                #pad the kernel with zeros to same length as signal
                k = np.hstack((k, np.zeros(len(y) - len(k))))

            # """roll kernel so that peak is centered"""
            # kernelmax = np.where(k == max(k))[0][0]
            # shift = int(k.size / 2) - kernelmax
            # k = np.roll(k, shift)
            # g = k  #use ZLP to deconvolve
        
        if kerneltype!=4:
            if algo == 'scipy':
                deconvolved_RL = restoration.richardson_lucy(y, g, iterations)
            elif algo == 'hyperspy':
                deconvolved_RL = deconv_function(y, kernel=g,iterations=iterations,psf_size=g.size)

            # plt.plot(xk, k, color= 'b',label = 'Raw')
            # plt.plot(xk, raw_deconvolved_RL, color= 'r',label = 'Raw_RL')
            # plt.plot(xg, g,label = "probe function") #type of kernel
            # plt.xlim([-100, 100])
            # plt.title('Raw data')
            # plt.legend()
            # plt.show()
            # plt.clf()

    else:
        if kerneltype == 1:
            s = FWHM / 2.3548
            g =  gaus(xg, 1, 0, s)
        elif kerneltype == 2:
            g = lor(xg, 1, 0, FWHM)
        elif kerneltype == 3:
            g = pvoigt(xg, 1, 0, FWHM, 0.5)
        elif kerneltype == 4:
            raise NotImplementedError('You have to give a ZLP file in order to accomplish kerneltype 4.')

    # print(y.shape,g.shape)
    'ZLP deconv'
    if algo == 'scipy':
        deconvolved_RL = restoration.richardson_lucy(y, g, iterations)
    elif algo == 'hyperspy':
        deconvolved_RL = deconv_function(y, kernel=g,iterations=iterations,psf_size=g.size)
    
    # plt.plot( x, y, color= 'b',label='Sub')
    # plt.plot( x, deconvolved_RL, label = 'Sub_RL')
    # plt.xlim(-100,100)
    # plt.ylim(0,0.0025)
    # plt.legend()
    # plt.show()
    # plt.clf()
    
    # iters = 100
    # RL_all = np.array([restoration.richardson_lucy(y, g, i) for i in range(1,iters+1)])
    # plt.imshow(np.flip(np.log(RL_all), axis = 0),cmap='jet',extent=[x[0], x[-1], 1, iters])
    # plt.show()
    # plt.clf()
    
    # ranges = [200, 1000]
    # RL_Raw_all = np.array([restoration.richardson_lucy(y, g, i) for i in range(1,iters+1)])
    # plt.imshow(np.flip(RL_Raw_all[:, ranges[0]:ranges[1]], axis = 0), cmap='jet', vmax = 0.01, extent=[x[0], x[-1], 1, iters])
    # plt.show()
    # plt.clf()
    return deconvolved_RL