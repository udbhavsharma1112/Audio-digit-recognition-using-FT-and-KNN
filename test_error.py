import os
import numpy as np
import librosa
import matplotlib.pyplot as plt



k_means = np.load('k_means.npy')
# Path to the test folder
folder_path = './test'

# Get a list of all .wav files in the test folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

# Initialize variables
prediction = np.zeros(100, dtype=int)
correct = 0

# Classification loop
for i in range(100):
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
    plt.plot(esd)
    plt.show()
    # Compare with k_means for classification
    check = np.zeros((10, 1000))

    for k in range(10):
        check[k, :] = (k_means[k, :] - esd)**2

    avg_check = np.abs(np.mean(check[:, :1000], axis=1))
    min_avg = np.min(avg_check)
    index = np.argmin(avg_check)

    prediction[i] = index

print(prediction)
# Calculate test error
for k in range(10):
    for i in range(10 * k, 10 * (k+1)):
        if prediction[i] == k:
            correct += 1

test_error = (100 - correct) / 100
print(f'Test Error: {test_error * 100}%')
