import os
import numpy as np
import librosa

# Path to the training folder
folder_path = './training'

k_means = np.load('k_means.npy')

# Get a list of all .wav files in the training folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

# Initialize variables
prediction = np.zeros(400, dtype=int)
correct = 0

# Classification loop
for i in range(400):
    file_path = os.path.join(folder_path, file_list[i])
    y, Fs = librosa.load(file_path, sr=None)

    # Calculate the Discrete Fourier Transform (DFT)
    dft_y = np.fft.fft(y)

    if len(y) % 2 == 0:
        dft_y = dft_y[:len(y) // 2]
    else:
        dft_y = dft_y[:((len(y) - 1) // 2) + 1]

    # Calculate the Energy Spectral Density (ESD)
    esd = np.abs(dft_y)**2
    esd = esd[:1000]

    # Compare with k_means for classification
    check = np.zeros((10, 1000))

    for k in range(10):
        check[k, :] = (k_means[k, :] - esd)**2

    avg_check = np.abs(np.mean(check[:, :1000], axis=1))
    min_avg = np.min(avg_check)

    print(min_avg)
    for i in range(10):
        if min_avg == avg_check[i]:
            index = i


    prediction[i] = index

# Calculate training error
for k in range(10):
    for i in range(40 * k, 40 * (k+1)):
        print("k" + str(k))
        print("prediction[i]" + str(prediction[i]))
        if prediction[i] == k :
            correct += 1

train_error = (400 - correct) / 400
print(f'Training Error: {train_error * 100}%')
