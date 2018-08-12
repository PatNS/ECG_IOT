# imports all together
import wfdb
import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import csv
import sys
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import time
from subprocess import call
from sklearn.neighbors import KNeighborsClassifier
import os
from socket import *

os.chdir("/home/pi/apneaecg")

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
    # Substitue mean for those outliers
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

os.chdir("/home/pi/apneaecg")
#filename = 'finalsvm.pkl'
filename = 'KNear.pkl'
classifier = pickle.load(open(filename, 'rb'))
	
os.chdir("/home/pi/apneaecg")

Repeat = 10;
while ( Repeat == 10):
	time.sleep (1)
	for root, dirs, files in os.walk(r'/home/pi/apneaecg/Incoming'):
		for file in files:
			if file.endswith('.csv'):
				ECGs = []
				with open('/home/pi/apneaecg/Incoming/'+file) as csvDataFile:
					csvReader = csv.reader(csvDataFile)
					for row in csvReader:
						ECGs.append(float(row[0]))
					t_ECG = np.linspace(0,180, 18000, endpoint=False) 
					TempSig = ECGs
					fps = 100    # ECG Signal sampling period is every 100Hz
					ECG = pd.DataFrame({'data' : TempSig}, index=t_ECG)
					filtered_ecg = butter_highpass_filter(ECG.data,20,fps)
					#Get HRs
					RRs, HRs = RRInterval(filtered_ecg)
					RRsCorrected = RRCorrection(RRs)
					HRsCorrected = RRCorrection(HRs)
					Df= GenFeatures(RRsCorrected, HRsCorrected)
					Dfarray = np.asarray(Df)
					DfarrayRS = Dfarray.reshape(1,-1)
					print("For file " + file + " the IOT Gateway prediction is: ")
					result = classifier.predict(DfarrayRS)
					print(result)
					if result == 'A':
						host = "ec2-34-242-160-43.eu-west-1.compute.amazonaws.com" # set to address of target computer
						port = 13000
						addr = (host, port)
						UDPSock = socket(AF_INET, SOCK_DGRAM)
						data = pickle.dumps(DfarrayRS,protocol=2)
						UDPSock.sendto(data, addr)
						UDPSock.close()
						#filename = "/home/pi/apneaecg/csvs/" + file + ".pkl"
						#pickle.dump(DfarrayRS, open(filename, 'wb'),protocol=2)			
						#cmd = "scp /home/pi/apneaecg/csvs/" + file + ".pkl Administrator@ec2-34-242-160-43.eu-west-1.compute.amazonaws.com:C:/Incoming/" + file + ".pkl"
						#call(cmd.split(" "))
					csvDataFile.close()
					os.remove('/home/pi/apneaecg/Incoming/'+file)
