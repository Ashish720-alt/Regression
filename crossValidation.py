from sklearn.model_selection import KFold
import numpy as np
from trainModel import trainLR

def kFoldCrossValidation(X_train, y_train, alphaValues, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Create k-fold splits
    alpha_errors = {}  # Dictionary to store mean error for each alpha
    
    for alpha in alphaValues:
        fold_errors = []  # Stores validation errors for this alpha
        
        for train_idx, val_idx in kf.split(X_train):
            # Split into training and validation sets for this fold
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Train on k-1 folds, validate on the kth fold
            _, val_error = trainLR(X_train_fold, X_val_fold, y_train_fold, y_val_fold, alpha, errorType = "mse", printErr = 0, printModel = 0)

            # Store the validation error
            fold_errors.append(val_error)

        # Compute the mean validation error over all k folds
        mean_val_error = np.mean(fold_errors)
        alpha_errors[alpha] = mean_val_error

    # Find the best alpha with the lowest mean validation error
    best_alpha = min(alpha_errors, key=alpha_errors.get)

    return best_alpha, alpha_errors[best_alpha]

