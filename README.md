# EMG signal processing
This project provides a collection of functions for processing EMG (electromyogram) signals. It includes various filters for noise reduction and frequency band separation, as well as functionality for generating visualizations of the processed signals.
# Instalation
To use this project, you need to have the following dependencies installed:
- NumPy
- pandas
- SciPy
- Matplotlib You can install these dependencies by running the following command: pip install numpy pandas scipy matplotlib
# Usage
The main functionality of this project is contained in the main function, which performs the processing of the EMG signal. Upon running the program, you will be prompted to select a filter or transformation to apply to the EMG signal. You can choose from the following options for transformations:
- Continuous Wavelet Transform (Key: CWT)
- Fourier Transform (Key: FT)
Filters that you can choose:
- High Pass Filter (Key: HPF)
- Band Pass Filter (Key: BPF)
- Median Filter (Key: MF)
- Low Pass Filter (Key: LPF)
- Rectify Signal Filter (Key: RSF)
After selecting a filter or transformation, the program will process the EMG signal accordingly and generate an image comparing the original and filtered signals. Additionally, it will generate plots showing the amplitude spectra for different frequency bands. To stop the program enter CL.
# Author
Tajra Selimović
