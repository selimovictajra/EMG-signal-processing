import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import cwt, morlet
from scipy import signal

# Load dataset from CSV
dataset = pd.read_csv('index_finger_motion_raw.csv', header=None)

# Extract the EMG signal column (adjust the column index if needed)
emg_signal = dataset.iloc[:, 0].values


def PlotOriginalEMG():
    plt.figure(figsize = (10, 7))
    plt.subplot(2, 1, 1)
    plt.plot(emg_signal)
    plt.ylim(-0.6, 0.6)
    plt.title('Original EMG Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')


def ContinuousWaveletTransform():
    # Plot the original EMG signal
    PlotOriginalEMG()

    # Define scales for the CWT
    scales = np.arange(1, 100)

    # Perform Continuous Wavelet Transform
    cwt_result = cwt(emg_signal, morlet, scales)

    # Plot the CWT result
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(cwt_result), extent=[0, len(emg_signal), scales[-1], scales[0]], cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title('Continuous Wavelet Transform')
    plt.xlabel('Sample')
    plt.ylabel('Scale')

    plt.tight_layout()
    plt.show()


def FourierTransform():
    # Perform Fourier Transform
    fourier_coeffs = np.fft.fft(emg_signal)
    frequencies = np.fft.fftfreq(len(emg_signal))

    # Plot the original EMG signal
    PlotOriginalEMG()

    # Plot the Fourier Transform result
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, np.abs(fourier_coeffs))
    plt.title('Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, np.max(frequencies))
    #plt.yscale('log')  # Set y-axis to logarithmic scale  # Adjust y-axis scale for better visualization

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()


def BandPassFilter():
    # Create bandpass filter for EMG
    high = 20 / (1000 / 2)
    low = 450 / (1000 / 2)
    b, a = signal.butter(4, [high, low], btype='bandpass')

    # Process EMG signal: filter EMG
    emg_filtered = signal.filtfilt(b, a, emg_signal)

    # Plot the original EMG signal and the filtered signal
    PlotOriginalEMG()

    plt.subplot(2, 1, 2)
    plt.plot(emg_filtered)
    plt.title('EMG Signal with Bandpass Filter')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.ylim(-0.7, 0.7)

    plt.tight_layout()
    plt.show()


def LowPassFilter():
    # Set the window size for the moving average filter
    window_size = 10

    # Apply the moving average filter to the EMG signal
    emg_filtered = np.convolve(emg_signal, np.ones(window_size) / window_size, mode='same')

    # Plot the original EMG signal and the filtered EMG signal
    time = np.arange(len(emg_signal))

    PlotOriginalEMG()

    plt.subplot(2, 1, 2)
    plt.plot(time, emg_filtered)
    plt.title('Filtered EMG Signal (Lowpass Filtered)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def MedianFilter():
    # Define the median filter parameters
    window_size = 7

    # Apply the median filter to the EMG signal
    emg_filtered = signal.medfilt(emg_signal, window_size)

    # Plot the original EMG signal and the median filtered signal
    time = np.arange(len(emg_signal))

    PlotOriginalEMG()

    plt.subplot(2, 1, 2)
    plt.plot(time, emg_filtered)
    plt.title('Median Filtered EMG Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim(-0.7, 0.7)

    plt.tight_layout()
    plt.show()


def HighPassFilter():
    # Set the window size for the moving average filter
    window_size = 10

    # Apply the moving average filter to the EMG signal
    emg_filtered = np.convolve(emg_signal, np.ones(window_size) / window_size, mode='same')

    # Compute the high-pass filtered signal by subtracting the low-pass filtered signal from the original signal
    emg_highpass = emg_signal - emg_filtered

    # Plot the original EMG signal, the low-pass filtered signal, and the high-pass filtered signal
    time = np.arange(len(emg_signal))

    PlotOriginalEMG()

    plt.subplot(2, 1, 2)
    plt.plot(time, emg_highpass)
    plt.title('Highpass Filtered EMG Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def RectifySignal():
    # Rectify EMG signal
    emg_rectified = abs(emg_signal)

    PlotOriginalEMG()

    plt.subplot(2, 1, 2)
    plt.title('Rectified EMG')
    plt.plot(emg_rectified)
    plt.ylim(-0.5, 0.7)
    plt.xlabel('Samples')
    plt.ylabel('EMG (a.u.)')

    plt.tight_layout()
    plt.show()


def main():
    print("In this programme you can perform a lot of transformations and filters on EMG signals.\n"
          "Transformation you can use are:\n"
          "   - Continuous Wavelet Transform (Key: CWT)\n"
          "   - Fourier Transform (Key: FT)\n"
          "Filters you can use are:\n"
          "   - High Pass Filter (Key: HPF)\n"
          "   - Band Pass Filter (Key: BPF)\n"
          "   - Median Filter (Key: MF)\n"
          "   - Low Pass Filter (Key: LPF)\n"
          "   - Rectify Signal Filter (Key: RSF)\n"
          "To close the programme enter CL")

    while True:
        key = input("\nEnter the wanted key: ")
        if key == 'CWT':
            ContinuousWaveletTransform()
        elif key == 'FT':
            FourierTransform()
        elif key == 'HPF':
            HighPassFilter()
        elif key == 'BPF':
            BandPassFilter()
        elif key == 'MF':
            MedianFilter()
        elif key == 'LPF':
            LowPassFilter()
        elif key == 'RSF':
            RectifySignal()
        elif key == 'CL':
            break
        else:
            print("The non-existent key entered! Please try again.\n")


if __name__ == '__main__':
    main()
