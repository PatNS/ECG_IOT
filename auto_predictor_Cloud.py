# imports all together
import wfdb
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import csv
import sys
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
import os
from socket import *

# Functions all together
def butter_highpass(cutoff, fs, order=15):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def RRInterval(SegmentIn):
    overThres = [] 
    mean = SegmentIn.mean()
    std = SegmentIn.std()
    Thres = mean + (2 * std)
    for i in SegmentIn:
        overThres.append(1*(i>Thres))
    overThresnp = np.array(overThres)
    changes = overThresnp[2:18000]-overThresnp[1:17999]
    ndx = []
    for i in range(17998):
        if changes[i] > 0:
            ndx.append(i+2)
    ndxlen=len(ndx)
    npndx =np.array(ndx)
    RRIntervals = npndx[2:ndxlen]-npndx[1:ndxlen-1]
    lenRR = len(RRIntervals)
    HRs = []
    for i in range(lenRR):
        HRs.append (6000/RRIntervals[i])
    return RRIntervals, HRs

def RRCorrection (RRIntervalValues):
	# Compare RRinterval values with mean+mean / 3 and mean-mean/3
	# Substitute mean for those outliers
    Anp = np.array(RRIntervalValues)
    Amean = Anp.mean()
    A = RRIntervalValues
    for c in range(len(A)):
       if A[c] > Amean+(Amean/3):
        A[c] = Amean
       if A[c] < Amean-(Amean/3):
        A[c] = Amean
    return A

def NN50 (RRintervals):
'''Calc NN50 measure (variant 1), defined as the number of pairs of adjacent RR- intervals 
where the first RR- interval exceeds the second RR- interval by more than 50 ms.
Note my calc is not strictly NN50 because the resolution doesn't show if it's 51ms only if it's 60ms greater
'''
    NN50_count = 0
    temp = 999
    for i in RRintervals:
        if (temp - i) > 5:
            NN50_count = NN50_count + 1
        temp = i
    return NN50_count

def NN50_count2 (RRint):
'''The NN50 measure (variant 2), defined as the number of pairs of adjacent RR-intervals where the 
second RR- interval exceeds the first RR interval by more than 50 ms
'''
    NN50_count2 = 0
    temp = 999
    for i in RRint:
        if (i - temp) > 5:
            NN50_count2 = NN50_count2 + 1
        temp = i
    return NN50_count2

def GenFeatures (RRsArray, HRsArray):
    data_features = []
    data_features.append(np.mean(HRsArray))
    data_features.append(np.std(HRsArray))
    data_features.append(NN50(RRsArray))
    data_features.append(NN50_count2(RRsArray))
    return data_features	


#svm = SVC(C=1.0, gamma='auto', kernel='rbf')
os.chdir("C:/Users/Administrator/Documents/apneanet")
#filename = 'finalsvm.pkl'
filename = 'RF_Class.pkl'
classifier = pickle.load(open(filename, 'rb'))
os.chdir("C:/Incoming")										

while True:
	# Data transfer using sockets
	host = ""
	port = 13000
	buf = 1024
	addr = (host, port)
	UDPSock = socket(AF_INET, SOCK_DGRAM)
	UDPSock.bind(addr)
	print "Waiting to receive messages..."
	count = 0
	while True:
		(data, addr) = UDPSock.recvfrom(buf)
		DfarrayRS = pickle.loads(data)
		result = classifier.predict(DfarrayRS)
		print(result)
		count = count + 1
		print(count)

# Below is the code used to load the pickle file when this method of data transfer is used
	'''time.sleep (1)
	for root, dirs, files in os.walk(r'C:\Incoming'):
		for file in files:
			if file.endswith('.pkl'):
				print(file)
				print("For file " + file + " the cloud system prediction is: ")
				DfarrayRS = pickle.load(open(file, 'rb'))
				result = classifier.predict(DfarrayRS)
				print(result)
				#csvDataFile.close()
				os.remove(file)'''
			