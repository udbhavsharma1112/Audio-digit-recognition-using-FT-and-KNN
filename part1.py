import librosa
import numpy as np
import matplotlib.pyplot as plt

# Read the audio file
file_path = './Training/9_01_33.wav'
y, Fs = librosa.load(file_path, sr=None)

# Compute the Discrete Fourier Transform (DFT)
dft_y = np.fft.fft(y)

# Keep only the positive frequencies
dft_y = dft_y[:len(y)//2] if len(y) % 2 == 0 else dft_y[:((len(y)-1)//2)+1]

# Compute the Power Spectral Density (PSD)
esd = np.abs(dft_y)**2
esd = esd[:1000]

# Plot the PSD
plt.subplot(3, 1, 1)
plt.plot(esd)
plt.show()
