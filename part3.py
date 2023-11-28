import numpy as np
import librosa
import matplotlib.pyplot as plt

k_means = np.load('k_means.npy')

# Read the audio file
file_path = './4digit.wav'
y, Fs = librosa.load(file_path, sr=None)

# Calculate duration in milliseconds
duration = int(len(y) / 48)
print(duration)
# Split into digits
c = 0
positions = []
means = np.zeros(duration)
plt.plot(y)
plt.show()
for i in range(duration):
    means[i] = np.mean(np.abs(y[48 * (i+1) - 47 : 48 * (i+1)]))

    if means[i] < 0.0001:
        c += 1
        positions.append(i)

s = 0
splits = np.zeros(c - 1)

for i in range(1, c):
    if abs(positions[i] - positions[i-1]) > 2 * 48:  # 2 msec
        s += 1
        splits[s] = positions[i]

splits = (splits[:s] * 48).astype(int)
indi_digi = np.zeros((s, np.max(splits)))

indi_digi[0, :splits[0]] = y[:splits[0]]

for i in range(1, s):
    indi_digi[i, :splits[i] - splits[i-1]] = y[splits[i-1] + 1 : splits[i] + 1]

# Classify digits
prediction = np.zeros(s, dtype=int)

for i in range(0,s):
    dft_digi = np.fft.fft(indi_digi[i, :])
    esd = np.abs(dft_digi)**2
    esd = esd[:1000]

    check = np.zeros((10, 1000))

    for k in range(10):
        check[k, :] = np.sqrt((k_means[k, :] - esd)**2)

    avg_check = np.abs(np.mean(check[:, :1000], axis=1))
    min_avg = np.min(avg_check)

    index = np.argmin(avg_check)
    prediction[i] = index

print(prediction)
