from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from PIL import Image
import pywt


SIGNALS_NAME = [
    "AbdoBelt",
    "AirFlow",
    "PPG",
    "ThorBelt",
    "Snoring",
    "SPO2",
    "C4A1",
    "O2A1",
]


Spectro_L = 90*4 # 4 pixels per seconds
WT_scales_number = 8 # number of scales to convolve with 
# an image of a single 9000 signal is 360x8

def get_WT(X, k):
    print(k)

    WT = np.zeros((n_signal, len(scales), Spectro_L))
    for i in range(n_signal):
        coef, _ = pywt.cwt(X[i], scales, "morl")
        coef = np.sqrt(np.abs(coef)) * np.sign(coef)
        img = Image.fromarray(coef)
        img = img.resize((Spectro_L, coef.shape[0]))
        coef = np.array(img)

        WT[i, :, :] = coef

    return WT, k


def normalize_dataset_per_patient(data):
    num_patient = data[:, 1]

    data = data[:, 2:]
    data = data.reshape((data.shape[0], 8, 9000))
    norm_data = np.zeros(data.shape)
    
    patients = np.unique(num_patient)

    for p in patients:
        patient_data = data[num_patient==p]
        mean0 = np.mean(patient_data, axis = 2)
        std0  = np.std(patient_data, axis = 2)
        patient_data = (patient_data - mean0[:,:,np.newaxis])/std0[:,:,np.newaxis]
        norm_data[num_patient==p] = patient_data
    return norm_data


if __name__ == "__main__":

    hf_wt = h5py.File("WT_train.h5", "w")

    PATH_TO_TRAINING_DATA = "additional_files_dreem/X_train.h5"
    PATH_TO_TRAINING_TARGET = "y_train_tX9Br0C.csv"
    h5_file = h5py.File(PATH_TO_TRAINING_DATA)

    norm_data = normalize_dataset_per_patient(h5_file['data'])

    scales = np.logspace(0, np.log10(500), WT_scales_number)
    print(scales)

    N_sample, T_size = h5_file["data"].shape
    n_signal = len(SIGNALS_NAME)
    T_size = T_size // n_signal

    pool = Pool(cpu_count())
    print(cpu_count())
    results = [get_WT(x, k) for k, x in enumerate(norm_data)]
    results = [results[r[1]][0] for r in results]
    results = np.array(results)

    print(results.shape)
    pool.close()

    hf_wt.create_dataset('data', data=results)
    hf_wt.close()
