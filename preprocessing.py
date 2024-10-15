# Importing numpy 
import numpy as np
# Importing Scipy 
import scipy as sp
from skimage.restoration import denoise_wavelet
from scipy.signal import savgol_filter
from scipy.signal import medfilt

#band pass filter between 0.5 and 40 hz
from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def median(signal, kernel_size=3):# input: numpy array 1D (one column)
    array=np.array(signal)   
    #applying the median filter
    med_filtered=sp.signal.medfilt(array, kernel_size=kernel_size) # applying the median filter order3(kernel_size=3)
    return  med_filtered # return the med-filtered signal: numpy array 1D
#notch filter apllied at 50hz
def Implement_Notch_Filter(time, band, freq, ripple, order, filter_type, data):
    # time:   Time between samples
    # band:   The bandwidth around the centerline freqency that you wish to filter
    # freq:   The centerline frequency to be filtered
    # ripple: The maximum passband ripple that is allowed in db
    # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
    #         IIR filters are best suited for high values of order.  This algorithm
    #         is hard coded to FIR filters
    # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
    # data:         the data to be filtered
    from scipy.signal import iirfilter
    fs   = 256#1/time
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def filter_teeth(x):
    x=median(x)
    x=savgol_filter(x, 30, polyorder=5 ,mode='nearest')


    # x = butter_bandpass_filter(x, .1, 10, 256, 3)
    # x = butter_bandpass_filter(x, .1, 10, 256, 3)
    # x = butter_bandpass_filter(x, .1, 10, 256, 3)
    # x = butter_bandpass_filter(x, .1, 10, 256, 3)

    # x=Implement_Notch_Filter(None, band=.1, freq=1, ripple=100, order=2, filter_type='butter', data=x)
    # x=Implement_Notch_Filter(None, band=.1, freq=.5, ripple=100, order=2, filter_type='butter', data=x)
    # x=Implement_Notch_Filter(None, band=2, freq=4, ripple=100, order=2, filter_type='butter', data=x)
    # x=Implement_Notch_Filter(None, band=2, freq=6, ripple=100, order=2, filter_type='butter', data=x)
    # x=Implement_Notch_Filter(None, band=2, freq=7.1, ripple=100, order=2, filter_type='butter', data=x)
    # x=Implement_Notch_Filter(None, band=2, freq=8.2, ripple=100, order=2, filter_type='butter', data=x)
    # x=Implement_Notch_Filter(None, band=2, freq=5, ripple=100, order=2, filter_type='butter', data=x)

    # fs = 256
    # lowcut = 20
    # highcut = 49
    # # lowcut = 0.20 * 128
    # # highcut = 0.30 *128
    # x = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    # x = median(x, 9)
    # x = savgol_filter(x, 10, polyorder=5 ,mode='nearest')
    # x = median(x)
    # x = savgol_filter(x, 30, polyorder=5 ,mode='nearest')

    return x

def filter_eyebrows(x,param=[11,3,12]):
    # fs = 256
    # med_size,lowcut,highcut = param[0],param[1],param[2]
    # x=butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    # x=median(x, med_size)
    # x=savgol_filter(x, 10, polyorder=5 ,mode='nearest')

    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)

    x= Implement_Notch_Filter(None, band=10, freq=18, ripple=100, order=2, filter_type='butter', data=x)
    x= Implement_Notch_Filter(None, band=10, freq=20, ripple=100, order=2, filter_type='butter', data=x)

    x = savgol_filter(x, 3, 2)
    x = denoise_wavelet(x,method='BayesShrink',mode='soft',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)

    return x

def filter_right(x):
    # fs = 256
    # x=median(x)
    # x=butter_bandpass_filter(x, lowcut=0.5, highcut=30, fs=fs, order=2)
    # x=denoise_wavelet(x, method='BayesShrink',mode='hard',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)
    # x=savgol_filter(x, 120, polyorder=3,mode='constant')

    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)

    x= Implement_Notch_Filter(None, band=10, freq=18, ripple=100, order=2, filter_type='butter', data=x)
    x= Implement_Notch_Filter(None, band=10, freq=20, ripple=100, order=2, filter_type='butter', data=x)

    x = savgol_filter(x, 3, 2)
    x = denoise_wavelet(x,method='BayesShrink',mode='soft',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)

    return x

def filter_left(x):
    # fs = 256
    # lowcut = 0.5
    # highcut = 5
    # x=median(x)
    # x=butter_bandpass_filter(x, lowcut, highcut, fs, order=2)
    # # x=denoise_wavelet(x,method='BayesShrink',mode='soft',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)
    # x=savgol_filter(x, 20, polyorder=5 ,mode='nearest')

    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)

    x = Implement_Notch_Filter(None, band=10, freq=18, ripple=100, order=2, filter_type='butter', data=x)
    x = Implement_Notch_Filter(None, band=10, freq=20, ripple=100, order=2, filter_type='butter', data=x)

    x = savgol_filter(x, 3, 2)
    x = denoise_wavelet(x,method='BayesShrink',mode='soft',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)
    return x

def filter_both(x):
    # fs = 256
    # med_size,lowcut,highcut = 11,3,12
    # x=butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    # x=median(x, med_size)
    # x=savgol_filter(x, 10, polyorder=4 ,mode='nearest')

    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)
    x = butter_bandpass_filter(x, 0.1, 10, 256, 3)

    x= Implement_Notch_Filter(None, band=10, freq=18, ripple=100, order=2, filter_type='butter', data=x)
    x= Implement_Notch_Filter(None, band=10, freq=20, ripple=100, order=2, filter_type='butter', data=x)

    x = savgol_filter(x, 3, 2)
    x = denoise_wavelet(x,method='BayesShrink',mode='soft',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)

    return x