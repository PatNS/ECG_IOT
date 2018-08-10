# ECG_IOT
An IOT 2 stage prediction system using ECG signals for sleep apnea detection with Raspberry Pi gateway and Amazon Web Services (AWS) cloud architecture

![ECG Screening Architecture](https://github.com/PatNS/ECG_IOT/blob/master/ECGScreeningArchitecture.png "ECG Screening Architecture") 

ECG Data Pre-Processing and Classifier Training.ipynb
This jupyter notebook was written using Python version 2.7. It includes the code for the pre-processing and feature extraction of the ECG signal files from PhysioNet apnea-ecg dataset (https://www.physionet.org/physiobank/database/apnea-ecg/). It also includes the classifier training and testing using the features extracted from this data.

auto_predictor_Laptop.py
This python script is a predicition program that is run from a windows PC with Python version 2.7. It can predict from 3 minute ECG signal segments if an apnea event is occurring according to the ECG signal features in that segment. It sends the segments which it predicts as having sleep apnea to an AWS (https://aws.amazon.com/what-is-cloud-computing/?sc_channel=EL&sc_campaign=amazonfooter) cloud server which is running a second stage prediciton program.

auto_predictor_Cloud.py
This python script is a predicition program that is run on an AWS cloud windows server with Python version 2.7. It receives signal feature lists from a 1st stage apnea prediction screener running on a Raspberry Pi via a sockets network transfer. Using these feature lists it makes a prediciton if sleep apnea is occurring.

auto_predictor_RaspberryPi.py
This python script is a predicition program that is run from a Raspberry Pi 3 with Python version 3. It can predict from 3 minute ECG signal segments if an apnea event is occurring according to the ECG signal features in that segment. It sends the features it calculates from any signal segment, which it predicts as having sleep apnea, to an AWS cloud server which is running a second stage prediciton program.
