import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Path to the training folder
folder_path = './training/'

# Get a list of all .wav files in the training folder
file_list = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

# Initialize the feature set
feature_set = np.zeros((len(file_list), 1000))
print(len(file_list))
# Compute ESD feature set for training data
for i, file in enumerate(file_list):
    file_path = os.path.join(folder_path, file)
    y, Fs = librosa.load(file_path, sr=None)
    dft_y = np.fft.fft(y)

    if len(y) % 2 == 0:
        dft_y = dft_y[:len(y) // 2]
    else:
        dft_y = dft_y[:((len(y) - 1) // 2) + 1]

    esd = np.abs(dft_y)**2
    feature_set[i, :] = esd[:1000]
    plt.plot(esd[:1000])
    plt.show()
# K-means clustering
k_means = np.zeros((10, 1000))

for k in range(10):
    for i in range(1000):
        average = np.sum(feature_set[(40 * (k)):(40 * (k+1)), i])
        k_means[k, i] = average / 40

fig, axes = plt.subplots(5, 2, figsize=(10, 12))

for k in range(10):
    row, col = divmod(k, 2)
    axes[row, col].plot(k_means[k])
    axes[row, col].set_title(f'Cluster {k}')

plt.tight_layout()
plt.show()


np.save('k_means.npy', k_means)
