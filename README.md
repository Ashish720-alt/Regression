# Regression
Predicting Deaths Caused by Cancer

# Run
Run python3 main.py on terminal

# List of possible metrics which can be used:
Key Metrics:
Regression: RMSE, MAE, R2-score.
Classification: Accuracy, precision, recall, F1-score, AUC-ROC.

# Pipeline Algorithm of main.py
Step 1: Split the Dataset
    Divide the data into:
        1. Training set (60%): For training the model.
        2. Validation set (20%): To tune hyperparameters and detect bias/variance.
        3. Test set (20%): For final performance evaluation after all tuning is complete.
    Note:
        Use stratified splitting if your dataset has imbalanced classes.
        For small datasets, you might consider cross-validation for more robust validation.

Step 2: Start with a Baseline Model
    Select a simple, interpretable baseline model. (For regression, start with linear regression (λ=0, degree = 1). For classification, consider logistic regression or a basic decision tree.)
    
    Train on the training set and evaluate:
        1. Training error: How well the model fits the training data.
        2. Validation error: How well the model generalizes to unseen data.
    Use this baseline to establish a reference point for further improvements.

Step 3: Cross-Validation for Hyperparameter Tuning
    Use k-fold cross-validation on the training set to find optimal hyperparameters:

    Hyperparameters to Tune:
        1. Regularization strength (λ): Control overfitting by penalizing large coefficients.
        2. Model complexity (e.g., polynomial degree): Adjust the function's flexibility.
        Steps:
            1. Split the training set into k-folds (e.g., k=5).
            2. Train on k−1 folds and validate on the remaining fold.
            3. Rotate through all folds, compute average validation error for each hyperparameter combination.
    Output: Best combination of hyperparameters (e.g., λ=0.1, degree = 3).

Step 4: Validate on the Validation Set
    Train the model using the best hyperparameters on the entire training set.

    Evaluate the model on the validation set:
        If validation error is high, diagnose whether it’s due to bias or variance:
            1. High bias (underfitting): Training and validation errors are both high.
        Solution: , increase degree, or add features.
        
            2. High variance (overfitting): Training error is low, but validation error is high.
        Solution: Increase λ, reduce degree, or remove features.
    
    If high validation error, you make minor adjustments in this step:
        1. If high bias: Decrease λ (and other hyperparameters) by small amounts and again restart from step 4. If it works,
           choose this and move on to step 6 else step 5.
        2. If high variance: Increase λ (and other hyperparameters) by small amounts and again restart from step 4. If it works,
           choose this and move on to step 6 else step 5.

Step 5. If high validation error, make major changes to model.
    If high validation error only, then:
        1. Large λ Coefficient Changes
            a. If high bias - consider drastically lower values of λ, and repeat from step 3.
            b. If high variance - consider drastically higher values of λ, and repeat from step 3. 

         Note that all steps ask you to repeat from step 3 i.e. you do k-fold cross-validation to find best combination of other hyperparameters (none here), then train on whole training dataset and check on validation set. If still high validation error, (hopefully same type as before like earlier high bias and still high bias) then move on to part 2 of step 5 and discard any changes made in this step. But if it works, ignore subsequent steps

        2. If High Variance only, Optimize Training Set Size
            If high variance only and 5.1 fails, take increasing size subsets of the training data, and for each subset plot the training and 
            validation errors. Then 2 cases:
            a. Large gap between training and validation errors forall/majority subsets of training data - Add more data to training data i.e.
               refactor dataset into 70:10:20 as training, validation and test data. Then do cross validation from step 3, and then step 4 (without
               minor tuning if still high validation error). if it works move to step 6 else discard everything (i.e. reset training set size) and
               move on to step 5.3
            b. Small gaph between training and validation curves - Simply move on to step 5.3, training set size doesnt influence variance here.
        
        3. Optimize features
            a. If high bias, do feature engineering (interaction terms or polynomial terms) to get more features. Go back to step 3 and then step 4 without minor tuning; if works move on to
                step 6, else move on to step 6, and note that model not trained well enough.
            b. If high variance, remove some features using domain knowledge. Go back to step 3 and then step 4 without minor tuning; if works move on to step 6, else move on to step 6, and note that model not trained well enough.

Step 6: Evaluate on the Test Set
    If model doesn't perform well, say training model failed and give reason.

    Once the model performs well on the validation set:
        Train it on the entire training + validation set using the best hyperparameters.
    Evaluate final performance on the test set.


# A note: Variants of Linear Regression for Modelling Real Valued Functions.

1. Ridge regression or Tikhanov Regression.
   Note that in standard linear regression with regularization, the cost function is given by || θ X - y ||_2^2 + α ||  θ ||_2^2 , where X is the 
   data matrix where a column represents a single datapoint, θ is the learned parameters as a row vector and α is the regularization parameter.

   Then ridge regression has the cost function || θ X - y ||_2^2 + || M θ ||_2^2 , where M is a matrix called Tikhanov matrix. If you take M = α I, then
   you get standard regression also called L2 regularization.

2. Lasso Regression.
   This is simply L1 regularization i.e. the cost function is || θ X - y ||_2^2 + α || θ ||_1

3. Polynomial Regression.
   Here the cost function is the same as L2-regularization however the model is no longer y_pred = θ x, instead it is:

   For some natural m, let X'(xi) be defined as the column matrix X'(xi) = [1, xi, xi^2, xi^3, ... xi^m ]^T. Then if β_j is any row vector of learnable
   parameters, the model assumed is:
   y =  ∑_i β_i * X'(xi)

   (Note that the polynomial terms are only in the same variable, no cross-polynomial terms.)