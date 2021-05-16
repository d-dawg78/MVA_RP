import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
from tqdm import trange

plt.rcParams["figure.figsize"] = (18, 2)


def normalize_dim(data):
    """
    Function for normalizing 1D signals across 8 dimensions.
    """
    for y in trange(data.shape[1]):
        mean = np.mean(data[:, y, :])
        std  = np.std(data[:, y, :])
        data[:, y, :] = (data[:, y, :] - mean) / std
        
    return data


def normalize_ppson(data):
    """
    Function for normalizing 1D signals independently for each set of 8 signals.
    """
    for x in trange(data.shape[0]):
        for y in range(data.shape[1]):
            mean = np.mean(data[x, y, :])
            std  = np.std(data[x, y, :])
            data[x, y, :] = (data[x, y, :] - mean) / std
        
    return data


def compute_spectrograms(data, status='log', verbose=False):
    """
    Compute spectrogram according to given parameters.
    Status:
    -- log:  log spectrogram before standardization 
    -- std:  regular standardization
    -- norm: normalization
    """
    n_fft=11
    hop_length=10
    win_length=10
    
    # Change 100 and 90 depending on chosen n_fft, hop_length, and win_length
    result = np.zeros((data.shape[0], data.shape[1], 8, 900))
    
    if (status == 'log'):
        for x in trange(data.shape[0]):
            for y in range(data.shape[1]):
                
                # Compute power spectrum
                spectrogram_librosa = np.log(np.abs(librosa.stft(data[x, y, :], \
                        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')) ** 2)
                                
                spectrogram_librosa[:, :] = (spectrogram_librosa[:, :] - np.mean(spectrogram_librosa)) / \
                        np.std(spectrogram_librosa)
                
                result[x, y, :, :] = spectrogram_librosa[:, :]
    
    elif (status == 'std'):
        for x in trange(data.shape[0]):
            for y in range(data.shape[1]):
                
                # Compute spectrogram of size (100 * 90)
                spectrogram_librosa = np.abs(librosa.stft(data[x, y, :], \
                        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')) ** 2
                
                spectrogram_librosa[:, :] = (spectrogram_librosa[:, :] - np.min(spectrogram_librosa)) / \
                        (np.max(spectrogram_librosa) - np.min(spectrogram_librosa))
                
                result[x, y, :, :] = spectrogram_librosa[:, :]
                
    elif (status == 'norm'):
        for x in trange(data.shape[0]):
            for y in range(data.shape[1]):
                
                # Compute spectrogram of size (100 * 90)
                spectrogram_librosa = np.abs(librosa.stft(data[x, y, :], \
                        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')) ** 2
                
                spectrogram_librosa[:, :] = (spectrogram_librosa[:, :] - np.mean(spectrogram_librosa)) / \
                        np.std(spectrogram_librosa)
                
                result[x, y, :, :] = spectrogram_librosa[:, :]
                
    elif (status == 'mel'):
        for x in trange(data.shape[0]):
            for y in range(data.shape[1]):
                
                spectrogram_librosa = np.log(librosa.feature.melspectrogram(y=data[x, y, :],sr=100,S=None, \
                        n_fft=n_fft,hop_length=hop_length,win_length=win_length,window='hann',n_mels=8,
                        fmax=50))
                
                spectrogram_librosa = (spectrogram_librosa - np.mean(spectrogram_librosa)) / \
                        np.std(spectrogram_librosa)
                
                if ((x == 0 or x == 18 or x == 52) and verbose == True):
                    plt.imshow(spectrogram_librosa, interpolation='nearest', aspect='auto', cmap ='coolwarm')
                    plt.title('X = {}, Y = {}'.format(x, y))
                    plt.show()
                
                result[x, y, :, :] = spectrogram_librosa[:, :]    
                
    else:
        print("Spectrogram status has not been selected. The possibilities are log, std, and norm.")
        
    return result