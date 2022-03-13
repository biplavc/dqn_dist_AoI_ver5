## https://stackoverflow.com/questions/55700338/how-to-generate-a-complex-gaussian-white-noise-signal-in-pythonor-numpy-scipy

import numpy as np
import math 
# import matplotlib.pyplot as plt
# from matplotlib import mlab



def getSNR(loc1, loc2):
    np.random.seed(42)
    distance = math.sqrt((loc1[0]-loc2[0])**2 + (loc1[0]-loc2[0])**2)
    x = np.random.normal(loc=0, scale=math.sqrt(0.99), size=(1, 2)).view(np.complex128) ## estimated 
    y = np.random.normal(loc=0, scale=math.sqrt(0.01), size=(1, 2)).view(np.complex128) ## error 
    z = x+y
    h = abs(z[0])**2

    power = 1 ## watts ## 40 DBM = 10 WATT among 10 RBs
    gain = h
    loss = distance**(-3.76)
    noise_power = 10**(-10)
    snr = np.round(10*math.log10(power*gain*loss/noise_power),2)
    return snr