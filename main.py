import pandas as pd
from trainModel import trainLR
import features
from dataManipulation import sanitize_data, extract_features_and_target, splitdata, normalize
from crossValidation import kFoldCrossValidation
from error import isErrorHigh
from hyperParameterChange import smallPercents, relativelyChanged
import numpy as np


''' Step 0. Preprocessing the dataset '''
# Load the dataset from .csv file 
datasetCSVFile = './cancer-regression/cancer_reg.csv'
data =  sanitize_data(pd.read_csv(datasetCSVFile) )  #sanitize_data takes care of NaN, inf, and -inf values

# # See what data looks like:
# print(data.head(4))
# print(data.iloc[0])

# Extract features and target variable, for features given in feature.py
X, y = extract_features_and_target(data, features.SELECTED_FEATURES, features.TARGET_VARIABLE)

''' Step 1. Split the dataset into training, validation and test data; normalize features if necessary '''

# Split in the ratio 60:20:20 by default, can change ratios by passing optional parameter splitRatio
X_train, X_val, X_test, y_train, y_val, y_test = splitdata(X, y)

# Normalize the features using Zscore normalization by default, can be changed by passing optional parameter normalization
X_train_scaled, X_val_scaled, X_test_scaled = normalize(X_train, X_val, X_test)

''' Step 2. Train a baseline Model.'''

print("Baseline Training...")
# Train Ridge Regression Model for given regularization constant alpha and calculate MSE as metric (can change metric)
alpha = 1 #alpha = 1 is accepted as moderate regularization constant, 0.1 is considered low and 10 is considered high.
baselineTrainErr, baselineValErr = trainLR(X_train_scaled, X_val_scaled, y_train, y_val, alpha)


''' Step 3. Use k-fold cross validation for finding optimal hyperparameter configurations.'''

k = 5
alphaValues = [0.01, 0.1, 1, 10 , 100]
alphaBest, alphaBestError = kFoldCrossValidation(X_train, y_train, alphaValues, k)

# # See what is best alpha and errors for each alpha:
# print(alphaBest, alphaBestError )

''' Step 4.1) Train on whole training set using the best hyperparameter values and validate on validation set. 
         4.2) If high validation error, then check if high bias (underfitting) or high variance (overfitting). If any error, 
              try to make small changes and restart step 4 to check if acceptable ValError.
'''

print("Initial Validation Training...")
trainErr, ValErr = trainLR(X_train_scaled, X_val_scaled, y_train, y_val, alphaBest)


isHighBias = 0
isHighVariance = 0
if (isErrorHigh(ValErr, baselineValErr)): #isErrorHigh uses a tolerance factor of 1.1 by default, can change by optional parameter tolerance
    print("High Validation Error!")
    if ( isErrorHigh( trainErr, baselineTrainErr, tolerance=0.9 ) ): #Compare against a known threshold error 
        # to measure increase in bias or increase in variance and not generalization gap (for generalization gap compare against ValErr)
        print("\tHigh Training Error, hence high bias (underfitting)")
        isHighBias = 1
    else:
        print("\tAcceptable Training Error, hence high variance (overfitting)")
        isHighVariance = 1

isModelTrained = not (isHighBias or isHighVariance)

# Small Changes to hyperparameter configurations
if (not isModelTrained):
    print("\nAttempting minor tweaks to regularization parameter...")
    alpha, alphaError = alphaBest, alphaBestError
    for percent in smallPercents:
        newAlpha = relativelyChanged(alphaBest, percent, isHighVariance)

        _, newValErr = trainLR(X_train_scaled, X_val_scaled, y_train, y_val, newAlpha, printErr = 0, printModel = 0)
        
        if (newValErr < alphaBestError):
            alpha = newAlpha
            alphaError = newValErr

    isModelTrained ==  (not isErrorHigh(alphaError, baselineValErr))
    print("\tMinor tweaks" + ("" if isModelTrained else " didnt ") + "tune model!" )

print("")

''' Step 5. If model still untrained, do large changes in this order and head back to step 3 i.e. do cross validation after doing these large changes
    5.1) Large changes to hyperparameter
    5.2) Increase training set if high variance
    5.3) add features via feature engineering if high bias or remove features using domain knowledge if high variance
    
For step 5, we code only those steps we need.
'''

if (not isModelTrained):
    if (isHighBias):
        
        print("Attempting large changes to regularization parameter...")
        
        # Step 5.1
        alphaMin = min(alphaValues)
        percentValues = [0.01, 0.1, 1.0 , 10, 50, 100]
        newAlphaValues  = [0] + [ (percent * alphaMin) / 100 for percent in percentValues ]
        
        k = 5
        newAlphaBest, newAlphaBestError = kFoldCrossValidation(X_train, y_train, newAlphaValues, k)
        newTrainErr, newValErr = trainLR(X_train_scaled, X_val_scaled, y_train, y_val, newAlphaBest, printErr = 0, printModel = 0)
        
        if (isErrorHigh(newValErr, baselineValErr)):
            print("\tLarge regularization parameter changes didnt fix high bias.", '\n')
        else:
            alphaBest = newAlphaBest
            print("\tLarge regularization parameter changes fixes high bias with best alpha = ", alphaBest, '\n')
        


''' Step 6. Test data: Train with best hyperparameters on train+validation data, and find error on test data.'''

print("Training and Testing Final Model...")

finalTrainErr, TestErr = trainLR(np.concatenate([X_train_scaled, X_val_scaled], axis=0), X_test_scaled, 
                                            np.concatenate([y_train, y_val], axis=0), y_test, alphaBest, isVal=0)


