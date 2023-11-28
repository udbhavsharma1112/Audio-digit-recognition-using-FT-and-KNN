# DigitAudioRecognition
•In the drive, there are two folders test and training. The training folder contains speech samples recorded at a sampling rate of 48000 Hz.
 The duration of each .wav file is approximately 0.68s. Each .wav file contains the utterance of a digit. The nomenclature for file name is as 
 follows: digit_speakernumber_filenumber. For eg., the file with name “06_01_31” corresponds to the digit being uttered is 6, the speaker number
 is 01 and file number is 31. Download the two folders and write a matlab code for reading the speech files.
	
•Write a matlab code for generating the power spectral density based feature set for each digit. The feature set should be obtained by only using 
 the files in the training folder.    
•Use the feature set derived in (2) to report classification accuracy on both training set and test set.
•Download the file named “4digit.wav.” This file contains a speech with an utterance of 4 digits. Devise a strategy for segmenting the file into 4 
 segments such that each segment has only one one digit. Use the feature set derived in (2) to estimate the digit in each segment.
