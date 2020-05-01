
# coding: utf-8

# In[160]:


import numpy as np
import math 
import scipy
import scipy.signal

from itertools import combinations

from google.protobuf import text_format

import itertools

import logging
import os
import sys
import datetime

def whiteNoiseMostVoters(numberOfSources, starting_frequency = 0):
    ''' White noise localization 
    
     Input:
       - 
       - 
       - 
       - 
    
     Output:
       - 
    '''
    outpath = os.path.join('2_source',  '{:%Y%m%d_%H%M}'.format(datetime.datetime.now()))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    logfile = os.path.join(outpath, "log.txt")

    logging.basicConfig(filename=logfile, format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    f0 = starting_frequency #starting frequency index
    step = 1 #discretization (for now we want to use the 180 angles so we keep it like that)
    n_samples = 512
    # maybe to add - starting frequency (f0)
    Fn = n_samples/2. +  1 #number of frequencies in spectrogram 
    # noise in dB
    SNR = 20

    
    dev = 'lego' # Change if using Kemar or lego
    logger.info('Load transfer functions %s'%dev)
    
    H_theta_time_total = np.load('lego1_h_theta_time.npy')    
    
    Df = H_theta_time_total[:, ::step, 0].shape[1] #number of directions for a fine discretization
            
    anglesf = np.arange(Df, dtype=np.int64)*360./Df # list of angles in degrees

    
    runs = 50
    J = numberOfSources
    conf_matrix = np.zeros((360, 360)) #confusion  matrix
    
    #TO ADAPT for the runs value and maybe microphones value
    err_per_source = np.zeros((runs, J))
    min_err_per_source = np.zeros((runs, J))
    number_microphones = H_theta_time_total.shape[2]
    logger.info('Number of runs %s'%(runs))
    logger.info('Noise in decibel %s'%(SNR))
    logger.info('Number of sources %s'%(J))
    logger.info('Number of samples %s'%(n_samples))
    #To keep the minimal error between each source at each run 
    min_err_by_run = np.zeros(runs)

    for rns in range(runs):
        #choose the directions randomly on the fine grid
        #The first line only gives angles between 0-180
        #should we adapt it to be equivalent to the angles of our fine discretization ? probably not because it's index
        theta = np.random.choice(range(Df), numberOfSources, replace=False)
        
        Source_signal = np.zeros((n_samples, J)) #list of source signals
        for j in range(J):
            Source_signal[:, j] = np.random.randn(n_samples) #source in time: random gaussian signal 

        print(rns)
        theta_hat_all_microphones = np.zeros((number_microphones, J))
        
        #index of used microphone (only used when H_theta0_t = H_theta_total[:, :, m])
        for m in range(number_microphones):
            
            H_theta_time_one_mic = H_theta_time_total[:, :, m]

            H_theta_time_one_mic = H_theta_time_one_mic[:, ::step] #coarse discretization
            
            
            #We need to transpose because this matrix right now is time/direction and we need it to be direction/time    
            H_theta_freq_one_mic = np.fft.rfft(np.transpose(H_theta_time_one_mic), n_samples)
            #H_theta_freq_one_mic is direction/freq we might need to transpose it
            
            #coarse discretization,  model, no need of the ::step because already done in time 
            H_theta_freq_one_mic = np.transpose(H_theta_freq_one_mic)
            H_theta_freq_one_mic = H_theta_freq_one_mic[f0:,:] 
            [F, D] = H_theta_freq_one_mic.shape 
            #F: number of frequencies,  D: number of directions
            #print(F,D)
            
            #length of the convolution might have to adapt something
            obs_len = n_samples + H_theta_time_one_mic.shape[0] - 1


            #print(Source_signal)

            #recorded time domain signal
            yt = np.zeros((obs_len, )) 
            #print(Source_signal.shape, Source_signal)

            for j in range(J):
                #source signal convolved with corresponding directional response
                yt += np.convolve(Source_signal[:, j], H_theta_time_one_mic[:, theta[j]]) 

            # Generate noise at required SNR    
            sig_norm = np.linalg.norm(yt)
            noise_t = np.random.randn(obs_len, ) #additive gaussian noise
            noise_norm = sig_norm/(10.**(SNR/20.))
            noise_t = noise_norm*noise_t/np.linalg.norm(noise_t)

            yt += noise_t #noisy signal

            #y = stft(yt, Fn)[f0:, :] #spectrogram of recorded signal
            #print("Fn is ")
            #print(Fn)
            #print(yt.shape)
            (freq_samples, seg_times, y) = scipy.signal.stft(yt, Fn, nperseg=n_samples)
            N = y.shape[1] #number of frames
            #print(y.shape)

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
                Bj = np.abs(H_theta_freq_one_mic[:, d2])**2 #vectors in current guess
                #print(Bj.shape)
                Pj = Bj.dot(np.linalg.pinv(Bj)) #projection matrix
                #print(Pj.shape)

                proj_error = np.linalg.norm((np.eye(F) - Pj).dot(y_mean)) #projection error

                if proj_error <= smallest_norm:
                    smallest_norm = proj_error
                    best_ind = q2
                    best_dir = d2
            theta_hat_all_microphones[m] = step*np.array(best_dir) #map coarse index to fine index
        #print(theta_hat_all_microphones)
        #print(theta)
        theta_hat = np.median(theta_hat_all_microphones, axis=0)
        theta_hat = theta_hat.astype(int)
        #print(theta_hat)
        min_err, best_perm = calculate_angle_error(theta, theta_hat, anglesf) #calculate error between chosen and true directions
        conf_matrix[theta, best_perm] += 1
        
        min_err_by_run[rns] = min_err

        for src_j in range(J): #error per source
            err_per_source[rns, src_j] = np.sum(np.absolute(((best_perm[src_j]-theta[src_j]+180) % 360)-180))
            
        
        logger.info('Test %s, theta: %s, theta_hat: %s, err: %s'%(rns, anglesf[theta], anglesf[best_perm], min_err))
    logger.info('Err_per_source average: %s, median: %s, Min_err average: %s, median: %s'%(np.mean(err_per_source, axis=0), np.median(err_per_source, axis=0), np.mean(min_err_by_run, axis=0), np.median(min_err_by_run, axis=0)))
    logger.info('Err_per_source max: %s, min: %s, Min_err max: %s, min: %s'%(np.max(err_per_source, axis=0), np.min(err_per_source, axis=0), np.max(min_err_by_run, axis=0), np.min(min_err_by_run, axis=0)))

    
    return err_per_source

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

    
whiteNoiseMostVoters(2)
    

