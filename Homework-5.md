Homework 5
================
Jordan Nieusma
2023-03-30

## Set up the notebook

- Import packages (dplyr, randomForest, caret)

- Read in test and training data from the vowel dataset at
  <https://hastie.su.domains/ElemStatLearn/datasets>

## Goal: Understand and implement a random forest classifier.

## Task \#1:

Using the “vowel.train” data, develop a random forest (e.g., using the
“randomForest” package) for the vowel data, using all of the 11 features
and using the default values of the tuning parameters.

``` r
# Prepare train/test dfs for tasks
train$y <- factor(train$y)
test$y <- factor(test$y)

rf_default <- randomForest(y ~ ., data=train)
print(rf_default)
```

    ## 
    ## Call:
    ##  randomForest(formula = y ~ ., data = train) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 3.22%
    ## Confusion matrix:
    ##     1  2  3  4  5  6  7  8  9 10 11 class.error
    ## 1  48  0  0  0  0  0  0  0  0  0  0  0.00000000
    ## 2   1 47  0  0  0  0  0  0  0  0  0  0.02083333
    ## 3   0  0 48  0  0  0  0  0  0  0  0  0.00000000
    ## 4   0  0  0 47  0  1  0  0  0  0  0  0.02083333
    ## 5   0  0  0  0 46  1  0  0  0  0  1  0.04166667
    ## 6   0  0  0  1  0 41  0  0  0  0  6  0.14583333
    ## 7   0  0  0  0  1  0 45  2  0  0  0  0.06250000
    ## 8   0  0  0  0  0  0  0 48  0  0  0  0.00000000
    ## 9   0  0  0  0  0  0  1  0 47  0  0  0.02083333
    ## 10  0  0  0  0  0  0  1  0  0 47  0  0.02083333
    ## 11  0  0  0  0  0  1  0  0  0  0 47  0.02083333

## Task \#2:

Use 5-fold CV to tune the number of variables randomly sampled as
candidates at each split if using random forest, or the ensemble size if
using gradient boosting.

``` r
set.seed(1)

# Define seq of values representing no. of variables to try
mtry <- data.frame(mtry=c(1:10))

# Set train method to cross-validation w/5 folds, apply grid search
ctrl <- trainControl(method="cv", number=5, search="grid")

# Train using CV and print results
rf_cv <- train(y ~ ., data=train, method="rf", trControl=ctrl, tuneGrid=mtry)
print(rf_cv)
```

    ## Random Forest 
    ## 
    ## 528 samples
    ##  10 predictor
    ##  11 classes: '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 422, 425, 423, 421, 421 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    1    0.9677919  0.9645635
    ##    2    0.9678645  0.9646432
    ##    3    0.9602796  0.9562988
    ##    4    0.9450388  0.9395269
    ##    5    0.9469256  0.9416015
    ##    6    0.9336980  0.9270475
    ##    7    0.9374719  0.9311952
    ##    8    0.9241908  0.9165769
    ##    9    0.9261145  0.9186899
    ##   10    0.9148097  0.9062549
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

``` r
print(rf_cv$finalModel)
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 2.65%
    ## Confusion matrix:
    ##     1  2  3  4  5  6  7  8  9 10 11 class.error
    ## 1  48  0  0  0  0  0  0  0  0  0  0  0.00000000
    ## 2   1 47  0  0  0  0  0  0  0  0  0  0.02083333
    ## 3   0  0 48  0  0  0  0  0  0  0  0  0.00000000
    ## 4   0  0  0 47  0  1  0  0  0  0  0  0.02083333
    ## 5   0  0  0  0 46  1  0  0  0  0  1  0.04166667
    ## 6   0  0  0  0  0 45  0  0  0  0  3  0.06250000
    ## 7   0  0  0  0  2  0 44  2  0  0  0  0.08333333
    ## 8   0  0  0  0  0  0  0 48  0  0  0  0.00000000
    ## 9   0  0  0  0  0  0  1  0 46  1  0  0.04166667
    ## 10  0  0  0  0  0  0  1  0  0 47  0  0.02083333
    ## 11  0  0  0  0  0  0  0  0  0  0 48  0.00000000

## Task \#3:

With the tuned model, make predictions using the majority vote method,
and compute the misclassification rate using the ‘vowel.test’ data.

``` r
# Get X and y variables
X_test <- test[, -c(1)]
y_test <- test[, c(1)]

# Get predictions
pred_prob <- predict(rf_cv, newdata=X_test, type="raw")

# Print results
xtab <- table(pred_prob, y_test)
print(confusionMatrix(xtab))
```

    ## Confusion Matrix and Statistics
    ## 
    ##          y_test
    ## pred_prob  1  2  3  4  5  6  7  8  9 10 11
    ##        1  32  0  0  0  0  0  0  0  0  1  0
    ##        2   9 22  3  0  0  0  0  0  0 15  1
    ##        3   1 16 35  3  0  0  0  0  0  3  0
    ##        4   0  0  0 29  3  1  0  0  0  0  1
    ##        5   0  0  0  0 16  7  9  1  0  0  0
    ##        6   0  0  3  9 18 24  3  0  0  0  5
    ##        7   0  0  0  0  3  0 26  6  5  0  3
    ##        8   0  0  0  0  0  0  0 30  7  0  0
    ##        9   0  4  0  0  0  0  4  5 21  1 14
    ##        10  0  0  0  0  0  0  0  0  3 22  0
    ##        11  0  0  1  1  2 10  0  0  6  0 18
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5952          
    ##                  95% CI : (0.5489, 0.6403)
    ##     No Information Rate : 0.0909          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5548          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity           0.76190  0.52381  0.83333  0.69048  0.38095  0.57143
    ## Specificity           0.99762  0.93333  0.94524  0.98810  0.95952  0.90952
    ## Pos Pred Value        0.96970  0.44000  0.60345  0.85294  0.48485  0.38710
    ## Neg Pred Value        0.97669  0.95146  0.98267  0.96963  0.93939  0.95500
    ## Prevalence            0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
    ## Detection Rate        0.06926  0.04762  0.07576  0.06277  0.03463  0.05195
    ## Detection Prevalence  0.07143  0.10823  0.12554  0.07359  0.07143  0.13420
    ## Balanced Accuracy     0.87976  0.72857  0.88929  0.83929  0.67024  0.74048
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11
    ## Sensitivity           0.61905  0.71429  0.50000   0.52381   0.42857
    ## Specificity           0.95952  0.98333  0.93333   0.99286   0.95238
    ## Pos Pred Value        0.60465  0.81081  0.42857   0.88000   0.47368
    ## Neg Pred Value        0.96181  0.97176  0.94915   0.95423   0.94340
    ## Prevalence            0.09091  0.09091  0.09091   0.09091   0.09091
    ## Detection Rate        0.05628  0.06494  0.04545   0.04762   0.03896
    ## Detection Prevalence  0.09307  0.08009  0.10606   0.05411   0.08225
    ## Balanced Accuracy     0.78929  0.84881  0.71667   0.75833   0.69048
