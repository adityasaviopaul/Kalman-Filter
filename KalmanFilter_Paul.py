# -*- coding: utf-8 -*-
"""
Created on Sat Apr  20 02:41:57 2019
Author: Aditya Savio Paul

Course     : Space Systems (2018/2019)
Institute  : Institute of Technology
University : University of Tartu, Estonia

---------------------About---------------------------------------
The following code conceptualizes the approach of Kalman Filter in data prediction
and rationally encapsulates the predicted data and the true data. 

Outputs:
    Separate Plots for True , Predicted and Observational Values
    Single Plot with combined plots for all three.

Reference Websites :

Wiki
    https://en.wikipedia.org/wiki/Kalman_filter

GitHub
    https://github.com/savio23/Kalman-Filter

StackOverFlow
    https://stackoverflow.com/questions/16798771/numpy-dot-product    

-----------------------------------------------------------------
"""
#Importing Libraries

#numpy
import numpy as np
#matrix operations
from numpy.linalg import inv
#plotting library
import matplotlib.pyplot as plt
#scikit-image
from skimage import io
#CV Library
import cv2 as cv2
#skimage, Image Library
from skimage import io

R = 2000000             # Orbital radius
T = 7620                # Period of the orbit
PosStd = 500000         # Standard deviation of the position error
VelStd = 2000           # Standard deviaton of the velocity error
f = 0.1                 # Filtering frequency
w = 2 * np.pi / T       # Angular velocity
DataSize = int(T / f)   
Shape = (DataSize, 4)
Shape2 = (DataSize, 2)

#True State Positions for the given values
def true_state(t, r, p):
    x = r * np.cos(w * t)
    y = r * np.sin(w * t)
    x_dot =-r * w * np.sin(w * t)
    y_dot = r * w * np.cos(w * t)
    return np.array([x, y, x_dot, y_dot]).transpose()


TrueData = true_state(np.linspace(0, T, DataSize), R, T)
#Empty array for noise
Noise = np.zeros(Shape)

#Considering Noise
Noise[:, 0:2] = np.random.normal(loc=0.0, scale=PosStd, size=Shape2)
Noise[:, 2:4] = np.random.normal(loc=0.0, scale=VelStd, size=Shape2)
Observations = TrueData + Noise
dt = f

#Estimating Predictions
Predictions = np.zeros(shape=(DataSize, 4))
Predictions[0] = Observations[0]

F = np.array([[1, 0, dt,  0],                                       # State transition model
              [0, 1,  0, dt],
              [0, 0,  1,  0],
              [0, 0,  0,  1]])

B = np.array([[0.5 * dt**2,           0],                           # Control-input model
              [          0, 0.5 * dt**2],
              [         dt,           0],
              [          0,          dt]])

P = np.array([[PosStd**2,         0,         0,          0],        # Covariance matrix
              [        0, PosStd**2,         0,          0],
              [        0,         0, VelStd**2,          0],
              [        0,         0,         0,  VelStd**2]])       # Covariance of the observation noise

    
R = np.array([[PosStd**2,         0,         0,          0],
              [        0, PosStd**2,         0,          0],
              [        0,         0, VelStd**2,          0],
              [        0,         0,         0,  VelStd**2]])

H = np.identity(4)                                                  # Observation model, feel free to play with the values here


Q = np.array([[PosStd**2/4000000,         0,         0,          0],
              [        0, PosStd**2/4000000,         0,          0],
              [        0,         0, VelStd**2/4000000,          0],
              [        0,         0,         0,  VelStd**2/4000000]])



for i in np.arange(1, DataSize):
    x = Predictions[i - 1]                                          
    z = Observations[i]                                             
    u = -w**2 * x[0:2]
    
    #Implement a Kalman filter, I recommend using the wiki article and trying to write the math into the code, you need to implement the Prediction and Update steps

    
    #________Prediction Matrices_________#
    
    # State Estimate : Predicted
    x1 =F.dot(x) 
    x2 =B.dot(u)
    x_priori = x1 + x2
    #x_priori = F.dot(x) + B.dot(u)
                                       
    # Error Covariance : Predicted
    p1=F.dot(P).dot(F.T)
    P_priori = p1+Q                                   
    
    #_______Update Matrices_____________#
    
    # Measuring the Pre-Fit Residual
    y1 = H.dot(x_priori)
    y = z-y1
    
    # Measuring the Pre-Fit Covariance                                      
    
    S = R + H.dot(P_priori).dot(H.T)                               
    
    # Calculating  Optimal Kalman Gain

    K = P_priori.dot(H.T).dot(inv(S))                               
    
    # Updating State Estimate
    x_posteriori = x_priori+K.dot(y)                               
    
    # Updating Error Covariance
    I   = np.identity(len(K))
    
    P_posteriori = (I-K.dot(H)).dot(P_priori).dot(((I-K).dot(H)).T)+K.dot(R).dot(K.T)
    
    #P_posteriori = (I-K.dot(H)).dot(P_priori).dot((I-K.dot(H)).T)+K.dot(R).dot(K.T)
    
    #Measuring Post Fit Residual
    y = z-H.dot(x_posteriori)                                      
    
    #Iterative Variable Accumalator for State Estimate
    Predictions[i] = x_posteriori                               
    
    #Iterative Variable Accumalator for Error Covariance
    P = P_posteriori 


#Plotting and Displaying True and Predicted Kalman Filtered Values
    
plt.title("Kalman Filter Out")
plt.plot(TrueData[:, 0], TrueData[:, 1], label='True position')
plt.plot(Predictions[:, 0], Predictions[:, 1], label="Predictions") #Uncomment when you have the predictions
plt.plot(Observations[:, 0], Observations[:, 1], 'r.', label="Observations",markersize=0.05) #Uncomment to see the input




#Printing Separated Data Plots of True Position / Predicted Positons / Observations
fig1, (plot1,plot2,plot3)=plt.subplots(1, 3, sharey=True)
fig1.suptitle("True | Predictions | Observations", fontsize=16)

plot1.plot(TrueData[:, 0], TrueData[:, 1],'g-', label="True position")
plot2.plot(Predictions[:, 0], Predictions[:, 1], label="Predictions") 
plot3.plot(Observations[:, 0], Observations[:, 1], 'r.', label="Observations",markersize=0.05)

#Legend Plots
plot1.legend()
plot2.legend()
plot3.legend()
plt.legend()

#Showing Final Images / Plots
plt.show()


##Kalman Filter Reference Equations

#url1 = 'https://github.com/savio23/Kalman-Filter/blob/master/Basic_concept_of_Kalman_filtering.png'
#url2 = 'https://github.com/savio23/Kalman-Filter/blob/master/eqns.PNG'
##img1 = io.imread(url1)
#img2 = io.imread(url2)
##numpy_vertical = np.vstack((img1,img2))
#numpy_vertical_concat = np.concatenate((img1,img2), axis=1)
#cv2.imshow(numpy_vertical_concat)

#-----------------------------------------------END Of CODE-------------------------------------------------------------------------#


##--------------------------------------Beta Decoder Code_DoNotUncomment-------------------------------------------------------------##
#I  = np.identity(4)
#FT = np.transpose(F)
#HT = np.transpose(H)
#for i in np.arange(1, DataSize):
#    x = Predictions[i - 1]                                          
#    z = Observations[i]                                             
#    u = -w**2 * x[0:2]
  
#    x_priori = np.dot(F,x) + np.dot(B,u)
##    print(x_priori)
#    P_priori = np.add(np.dot(F,P),Q)
#    #print(P_priori)
#    y = np.subtract(z,np.dot(H,x))
#    #print(y)
#    S = np.add(R, np.dot(np.dot(H,P),HT))
#    #print(S)
#    SI = np.linalg.inv(S)
#    K = np.dot(P,HT,SI)
#    #print(K)
#    KI = np.linalg.inv(K)
#    x_posteriori = np.add(x,np.dot(K,y))
#    #print(x_posteriori)
#    T1=np.subtract(I,np.dot(K,H))
#    TI = np.linalg.inv(T1)
#    #print(TI)
#    P_posteriori = np.dot(np.subtract(I,np.dot(K,H)),P,np.add(np.subtract(I,np.dot(K,H)),np.subtract(np.dot(K,R),KI)))   
##    P_posteriori = (I[i]-(K[i]*H[i]))*P[i]*(I-K[i]*H[i])+K[i]*R[i]-KI[i]   
#    y = np.subtract(z,np.dot(H,x))
#    #print(y)
#    Predictions[i] = x_posteriori 
#    #print(Predictions)
#    P = np.subtract(I, np.dot(np.dot(K,H),P))
#    print(P)
#--------------------------------------------------------------------------------------------------------------------