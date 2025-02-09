
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split



def sanitize_data(data):
    defaultNanValue = 0
    defaultthresholdValue = 1e308
    defaultInfValue = 1e308
    
    # Define helper function to sanitize individual values
    def sanitize_value(value):
        if isinstance(value, (int, float, complex)):
            if np.isnan(value):
                return defaultNanValue
            elif np.isinf(value) or np.abs(value) > defaultthresholdValue:
                return defaultInfValue if value > 0 else -1 * defaultInfValue
        return value
    
    if isinstance(data, pd.DataFrame):
        # If data is a DataFrame, sanitize column-wise
        for column in data.columns:
            data[column] = data[column].apply(lambda val: sanitize_value(val))
    elif isinstance(data, np.ndarray):
        # If data is a NumPy array, sanitize element-wise
        for i in range(data.shape[1]):  
            data[:, i] = [sanitize_value(val) for val in data[:, i]]
    else:
        raise TypeError("Input data must be either a pandas DataFrame or a numpy ndarray.")
    
    return data

#extract features and target
def extract_features_and_target(data, FEATURE_NAMES, TARGET_NAME):
    X = data[FEATURE_NAMES]
    y = data[TARGET_NAME]
    return X, y


def splitdata(X, y, splitRatio = { "train": 0.6 , "val": 0.2, "test": 0.2 }):
    # Step 1: Split the data into train + validation and rest test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=splitRatio['test'], random_state=42)

    # Step 2: Split the train + validation into training and validation data
    conditionedSplit = splitRatio['val']/(splitRatio['train'] + splitRatio['val'])
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size= conditionedSplit, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test



#Use Zscore normalization i.e. x_i = (x - u)/σ  , where u and σ are calculated on the training data only.
def ZscoreNormalize(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

#Use min-max normalization i.e. x_i = (x - m)/[M - m]  , where m and M are the minimum and maximum values calculated on the training data only.
def MinMaxNormalize(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)  
    X_val_scaled = scaler.transform(X_val)      
    X_test_scaled = scaler.transform(X_test)    
    return X_train_scaled, X_val_scaled, X_test_scaled
    
#Use mean normalization i.e. x_i = (x - u)/[M - m]  , where u, m and M are the mean, minimum and maximum values calculated on the training data only.
def MeanNormalizer(X_train, X_val, X_test):  
    # Compute the mean, min, and max for each feature from the training data
    mean = np.mean(X_train, axis=0)
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    X_train_scaled = (X_train - mean) / (max_val - min_val)
    X_val_scaled = (X_val - mean) / (max_val - min_val)
    X_test_scaled = (X_test - mean) / (max_val - min_val)
    return X_train_scaled, X_val_scaled, X_test_scaled


def normalize(X_train, X_val, X_test, normalization = "standard"):  
    if (normalization == "standard" or normalization == "Zscore"):
        return ZscoreNormalize(X_train, X_val, X_test)
    elif (normalization == "minmax"):
        return MinMaxNormalize(X_train, X_val, X_test)
    elif (normalization == "mean"):
        return MeanNormalizer(X_train, X_val, X_test)
    elif (normalization == "none"):
        return X_train, X_val, X_test