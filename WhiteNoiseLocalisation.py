
# coding: utf-8

# In[55]:


import numpy as np
import math 
import scipy

from itertools import combinations

from google.protobuf import text_format

import itertools

import logging
import os
import sys
import datetime

def whiteNoise(numberOfSources, starting_frequency = 0):
    ''' White noise localization 
    
     Input:
       - 
       - 
       - 
       - 
    
     Output:
       - 
    '''
    outpath = os.path.join('output',  '{:%Y%m%d_%H%M}'.format(datetime.datetime.now()))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    logfile = os.path.join(outpath, "log.txt")

    logging.basicConfig(filename=logfile, format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    f0 = starting_frequency #starting frequency index
    step = 1 #discretization (for now we want to use the 180 angles so we keep it like that)
    #m = 1 #index of used microphone (only used when H_theta0_t = H_theta_total[:, :, m])
    n_samples = 256

    dev = 'lego' # Change if using Kemar or lego
    logger.info('Load transfer functions %s'%dev)
    
    H_theta_total = np.load('lego1_h_theta_time.npy')

    H_theta_stacked = H_theta_total[:, :, 0]
    for i in range(1, H_theta_total.shape[2]):
        H_theta_stacked = np.append(H_theta_stacked, H_theta_total[:, :, i],axis = 0)
        
    #H_theta0_t = H_theta_total[:, :, m] # For one mic
    H_theta0_t = H_theta_stacked
    
    H_theta_t = H_theta0_t[:, ::step] #coarse discretization
    Df = H_theta_t.shape[1] #number of directions for a fine discretization

    H_theta_t_transpose = np.transpose(H_theta_t)

    
    H_theta0 = np.fft.rfft(H_theta_t_transpose, n_samples) #load transfer functions (includes 4 microphones)
    H_theta0 = np.transpose(H_theta0)
    
    anglesf = np.arange(Df, dtype=np.int64)*360./Df # list of angles in degrees
    
    theta = np.random.choice(range(Df), numberOfSources, replace=False) #choose the directions randomly on the fine grid   
        
    H_theta = H_theta0[f0:, ::step] #coarse discretization,  model
    [F, D] = H_theta.shape #F: number of frequencies,  D: number of directions
    
    runs = 20
    J = numberOfSources
    conf_matrix = np.zeros((360, 360)) #confusion  matrix
    err_per_source = np.zeros((runs, J))
    min_err_per_source = np.zeros((runs, J))
    obs_len = n_samples + H_theta_t.shape[0] - 1#length of the convolution
    SNR = 20 # noise of 20 dB
    Fn = n_samples/2. +1 #number of frequencies in spectrogram 
    
    logger.info('Number of sources %s'%(J))
    
    for rns in range(runs):
        St_all = np.zeros((n_samples, J)) #list of source signals
        for j in range(J):
            St_all[:, j] = np.random.randn(n_samples) #source in time: random gaussian signal 

        theta = np.random.choice(range(Df), J, replace=False) #choose the directions randomly on the fine grid  

        yt = np.zeros((obs_len, )) #recorded time domain signal

        for j in range(J):
            yt += np.convolve(St_all[:, j], H_theta0_t[:, theta[j]]) #source signal convolved with corresponding directional response

        # Generate noise at required SNR    
        sig_norm = np.linalg.norm(yt)
        noise_t = np.random.randn(obs_len, ) #additive gaussian noise
        noise_norm = sig_norm/(10.**(SNR/20.))
        noise_t = noise_norm*noise_t/np.linalg.norm(noise_t)

        #yt += noise_t #noisy signal
        


        y = stft(yt, Fn)[f0:, :] #spectrogram of recorded signal
        N = y.shape[1] #number of frames

        y_mean = np.mean(np.abs(y)**2, axis=1) #mean power frame
        y_mean = y_mean/np.linalg.norm(y_mean) #normalize the observation

        # Exhaustive search algorithm

        # Initialize variables
        best_ind = np.inf #index corresponding to best direction tuple
        smallest_norm = np.inf #smallest projection error
        best_dir = theta #best direction tuple

        # Search all combinations
        pairs2 = combinations(range(D), J)
        for q2, d2 in enumerate(pairs2): 
            Bj = np.abs(H_theta[:, d2])**2 #vectors in current guess
            Pj = Bj.dot(np.linalg.pinv(Bj)) #projection matrix
            
            proj_error = np.linalg.norm((np.eye(F) - Pj).dot(y_mean)) #projection error

            if proj_error <= smallest_norm:
                smallest_norm = proj_error
                best_ind = q2
                best_dir = d2
        theta_hat = step*np.array(best_dir) #map coarse index to fine index
        min_err, best_perm = calculate_angle_error(theta, theta_hat, anglesf) #calculate error between chosen and true directions
        conf_matrix[theta, best_perm] += 1
        
        for src_j in range(J): #error per source
            err_per_source[rns, src_j] = np.sum(np.absolute(((best_perm[src_j]-theta[src_j]+180) % 360)-180));
            #min_err_per_source[rns, src_j] = min_err
        
        logger.info('Test %s, theta: %s, theta_hat: %s, err: %s'%(rns, anglesf[theta], anglesf[best_perm], min_err))
    
    #print(min_err_per_source)
    
    return err_per_source


# Taken from https://github.com/swing-research/scatsense/blob/master/core/signal.py
# Code written by Dalia El Badawy
def stft(x,F):
    '''Compute STFT of 1D real signal x
    F: number of frequencies
    '''
    wn = 2*(F-1) #window size in samples
    nhop = int(wn/2) #hop size, 50% overlap
    
    L = len(x) #length of input signal
    zero_pad = nhop #number of zeros to add before and after the signal
    rem = int(L%nhop) #number of samples leftover
    
    if rem>0: # adjust zero padding to have an integer number of windows
        zero_pad = int(wn-rem)#nhop-rem#wn-rem
    x = np.hstack((np.zeros((nhop,)),x,np.zeros((zero_pad,)))) #zero padding at the beginning to avoid boundary effects
    L = len(x) #new length of signal
    
    N = int((L-wn)/nhop + 1) #total number of frames
    X = np.zeros((int(F),N),dtype=np.complex128) #output STFT matrix
     
   
    w = 0.5- 0.5*np.cos(2*np.pi*np.arange(wn)/(wn)) #Hann window
    for i in range(N): #compute windowed fft for every frame
        xf = np.fft.rfft(x[int(i*nhop):int(i*nhop+wn)]*w)
        X[:,i] = xf

    return X 

# Taken from https://github.com/swing-research/scatsense/blob/master/core/signal.py 
# Code written by Dalia El Badawy
def calculate_angle_error(theta,theta_hat,angles):
    '''Average localization error in degrees (modulo 360)
    Also finds the best permutation that gives the lowest error
    Input:
        theta: array of true indices
        theta_hat: array of estimated indices
        angles: list of angles in degrees
    '''
    J = len(theta) #number of sources
    all_perms = itertools.permutations(theta_hat) #all permutations
    min_err = np.Inf

    for beta in all_perms:
        curr_err = np.sum(np.absolute(((angles[np.array(beta)]-angles[theta]+180) % 360)-180))*1./J;
        if curr_err<min_err:
            min_err = curr_err
            perm = np.array(beta)
            
    return min_err,perm

whiteNoise(1)

