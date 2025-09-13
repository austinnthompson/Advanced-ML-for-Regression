#!/usr/bin/env python
# coding: utf-8

# <h1>Helper Functions for Supervised Learning Analysis</h1>
# 

# # Preliminaries
# 
# This notebook contains functions we will use throughout the course. Most of these just aggregate, tabulate, or present results in a nice fashion. It helps to define these once, rather than having to write these lines of code for each notebook we do

# # Import Modules

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.metrics import log_loss,   accuracy_score,  precision_score, recall_score, f1_score, brier_score_loss
from sklearn.metrics import average_precision_score, roc_auc_score, cohen_kappa_score, classification_report , balanced_accuracy_score 
from sklearn.metrics import  make_scorer,  confusion_matrix, RocCurveDisplay, DetCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay  
from sklearn import tree


# # Cross-validation summary
# 
# Cross-validation is our primary method for evaluating models. The `cross_validate` function handles the cross-validation process, but it returns results for all the folds. The function here computes the mean metric for each metric provided. For example, this function can calculate the mean RMSE across all the folds in a regression setting.
# 

# In[2]:


#cv_results: output from a call to cross_validate
#metrics: metrics calculated in cross_validate. For example RMSE for regression or log-loss for classification
#metrics should be a list of strings specifying the metrics. The metrics should be listed as they appear in official sklearn metric documentation
#
#returns the mean metric for each metric in dict form
def cross_validation_summary(cv_results, metrics):

    results = {}

    #loop over metrics and compute mean quantity 
    for score in metrics:
        
        #get mean metric
        mean_metric = np.mean(cv_results['test_' + score])
        print(score, mean_metric)

        #store in dict
        results[score] = mean_metric 
        
    #return results as dict
    return results


# # Ridge/Lasso/ElasticNet Functions
# 
# We aim to perform similar analyses for Ridge, Lasso, and ElasticNet, including plotting relevant metrics against the penalty parameters and tuning hyperparameters. In this section, we define several functions to explore and optimize these models.
# 

# ## Plot Coefficients Against Penalty $\alpha$
# 
# With Lasso and Ridge, the coefficients shrink toward 0 as the penalty parameter $\alpha$ increases. The function below generates a plot that illustrates the evolution of each coefficient as $\alpha$ grows.
# 

# In[3]:


#plot ridge/lasso coefficients as a function of alpha
#reg_model: pipeline ridge or lasso model. reg_model['model'] is the ridge/lasso component of pipeline 
#alpha: list of penalty paramter alpha we wan to evaluate and compute coefficients
#X_train, y_train: training data
#
#returns nothing, plots alpha on x and coeficients on y for every predictor 
def plot_regularization_coefficients(reg_model, alphas,X_train, y_train):

    #store coefficients in list
    coefs = []
    #loop over all alphas
    for a in alphas:
        # Manually change alpha
        reg_model.set_params(model__alpha=a)
        # Fit the model with new alpha
        reg_model.fit(X_train, y_train)
        #store coefficients
        coefs.append(reg_model['model'].coef_)
    
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('coefficients')
    plt.title('Coefficients as a function of the penalty alpha')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axis('tight')
    plt.show()
 


# ## Plot Cross-Validation Error vs. $\alpha$
# 
# Next, we plot the average validation MSE against $\alpha$. For each value of $\alpha$, a cross-validation procedure (e.g., K-fold or leave-one-out) is performed. The average cross-validation error across these runs is then computed and plotted against $\alpha$.
# 

# In[4]:


#plot cv error (mse) vs alpha
#cv_errors: fpr each alpha the average MSE average over the runs in cross-validation procedure.
#alphas: list of penalty paramter alpha we wan to evaluate and compute cv error
#
#returns nothing, plots alpha on x and coeficients on y for every predictor
def plot_regularization_tuning(cv_errors, alphas):
    
    # Plot the CV error vs alpha
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, cv_errors, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.xlabel('Alpha (log scale)', fontsize=12)
    plt.ylabel('Mean Cross-Validation Error (MSE)', fontsize=12)
    plt.title('Cross-Validation Error vs. Alpha', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# ## 1-SE $\alpha$
# 
# The most logical choice for the best $\alpha$ is the one that minimizes the MSE. However, another commonly used approach in practice is the '1-SE' rule. This selects the largest $\alpha$ (i.e., a higher penalty, resulting in a simpler model) that produces an MSE within one standard error of the minimum MSE.  
# 
# To compute the standard error for a given $\alpha$, we calculate the MSE for each of the validation runs. We then compute the sample average and sample standard deviation of these  MSE values. The standard error is calculated as the standard deviation across the cross-validation runs divided by the square-root of the number of runs
# 
# Below, we provide the code to calculate the 1-SE $\alpha$.
# 

# In[5]:


#compute 1se alpha
#mean_errors: for each alpha there is a mean error 
#std_errors: standard error of the estimated MSE
#alphas: list of penalty paramter alpha we wan to evaluate and compute cv error

#
#returns teh 1SE alpha nad correspondign MSE 
def get_1SE_alpha(mean_errors,std_errors, alphas):
    
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(
        {'alpha': alphas,
        'mean': mean_errors,
        'std': std_errors}
    )
 
    # Find the best alpha and its corresponding mean error
    best_alpha_index =   df['mean'].idxmin()
    best_alpha =   df.loc[best_alpha_index, 'alpha']  
    best_mean_error =   df.loc[best_alpha_index, 'mean']
  
    # Compute the 1SE threshold
    threshold_error = best_mean_error + std_errors[best_alpha_index]
     
    # Find the largest alpha within the 1SE threshold
    #first filter to those below threshold
    filtered_df = df[df['mean'] < threshold_error]
    #get largest below threshold
    alpha_1se_index =  filtered_df['mean'].idxmax()
    alpha_1se =  df.loc[alpha_1se_index, 'alpha']    
    mean_error_1se = df.loc[alpha_1se_index, 'mean']  

    return alpha_1se, mean_error_1se


# # Hyperparameter Tuning Summary
# After performing hyperparameter tuning (using grid search or random search), this function aggregates the key results and stores them in a DataFrame. Notably, the returned DataFrame includes a list of the hyperparameter values and the corresponding mean metrics. For example, in the case of ElasticNet regression, it will display all values of $\alpha$ and $\rho$ along with the corresponding mean RMSE, mean MAE, etc. The DataFrame is then sorted by the metric used to optimize the hyperparameters (e.g., RMSE in regression contexts).
# 

# In[6]:


#cv_search: output from a call to a hyperparameter tuning function such as GridSearchCV or RandomSearchCV
#metrics: metrics calculated during cross validation. For example RMSE for regression or log-loss for classification
#metrics should be a list of strings specifying the metrics. The metrics should be listed as they appear in official sklearn metric documentation
#primary_metric: the metric the tuning function optimizes. For example use log-loss in a classification setting to determine best hyperparameters. refit must be 
#string in sklearn format
#
#returns a df. each row of df corresponds to a hyperparmeter combination.  The columns are the mean metrics 
def cv_results_aggregator(cv_search,metrics,primary_metric):

    #convert  cv hyperamater tuning results to data frame
    cv_df = pd.DataFrame(cv_search.cv_results_)

    #we only want to store the mean metric for each of the metrics of interest
    # Append 'mean_test_' to each string
    mean_metric_columns = ["mean_test_" + score for score in metrics]
    
    #only keep params and mean metric
    short_results = cv_df[['params'] + mean_metric_columns]
    
    # Sort by refit. most metrics we want to maximize so descebding gives us the best
    return short_results.sort_values(by='mean_test_' + primary_metric,ascending=False)


# # Best Hyperparameter Results
# 
# The `cv_results_aggregator` consolidates the results for all hyperparameters into a DataFrame. The next function presents only the results for the best hyperparameter, which corresponds to the first row of the DataFrame returned by `cv_results_aggregator`. The following code calls `cv_results_aggregator` and extracts the first row, appropriately renaming the labels so that the dictionary keys match the corresponding metrics.
# 

# In[7]:


#cv_search: output from a call to a hyperparameter tuning function such as GridSearchCV or RandomSearchCV
#metrics: metrics calculated during cross validation. For example RMSE for regression or log-loss for classification
#metrics should be a list of strings specifying the metrics. The metrics should be listed as they appear in official sklearn metric documentation
#primary_metric: the metric the tuning function optimizes. For example use log-loss in a classification setting to determine best hyperparameters. refit must be 
#string in sklearn format
#
#returns the mean metrics for best hyperperarmetr combination in dict format
def hypertuning_summary(cv_search,metrics,primary_metric):

    #get hyperparamter tuning results from cv_results_aggregator function 
    cv_df = cv_results_aggregator(cv_search,metrics,primary_metric)

    # Create the dictionary for first row results but modify the keys from column labels by dripping the mean test
    best_results_dict = {col.replace('mean_test_', ''): cv_df[col].iloc[0] for col in cv_df.columns}

    # Print each key-value pair on a new line
    for key, value in best_results_dict.items():
        print(f"{key}: {value}")

    #return results as dict
    return best_results_dict


 
 


# # Polynomial Transformer
# Creating a transformer in `sklearn` that performs only polynomial transformations without including interactions can be a bit tricky. Once we define the function below, we can pass it to `ColumnTransformer` to officially construct the transformer.

# In[8]:


#poly_features: list of featuers we want to take polynomial transformation of. Etiher strings or ints
#degree: what degree polynomial do we want (int)
#
##returns a list of polynomial transformation calls. We need to send this to *ColumnTransformer* to actually make the transformer
def polynomial_transform(poly_features, degree):
    poly_transformer = []
    #loop throuhg each feature and create a separate polynomial transfomer
    for col in poly_features:
        tmp= ('poly_'+ col,PolynomialFeatures(degree=degree, include_bias=False), [col])
        poly_transformer.append(tmp)

    return poly_transformer


# # Binary Classification Metrics

# This function prints (and returns) all the binary classification metrics we covered in the course. You would typically call it to evaluate your test set, or potentially a validation set.

# In[9]:


#y_true: actual class lables 
#pred_prob: predicted probabilities of each clas
#pred_class: predicted class label. note user could specify a non 0.5 threshold to determine this
#
# returns: prints (and returns) all classifcioant metrics: log-loss, AUC, accuracy, etc 
def compute_binary_class_metric(y_true, pred_prob, pred_class):
    #store results in dict
    metrics = {}

    #logloss
    metrics['logloss'] = log_loss(y_true, pred_prob)
 
    #  brier
    metrics['brier'] = brier_score_loss(y_true, pred_prob[:,1])
    
    #  aucroc
    metrics['roc_auc'] = roc_auc_score(y_true, pred_prob[:,1])
    
    #  aucprc
    metrics['pr_auc'] = average_precision_score(y_true, pred_prob[:,1])
    
    #accuracy   
    metrics['accuracy'] = accuracy_score(y_true, pred_class)

    # Precision, Recall, F1 Score
    metrics['precision'] = precision_score(y_true, pred_class, pos_label=1)
    metrics['recall'] = recall_score(y_true, pred_class, pos_label=1)
    metrics['f1'] = f1_score(y_true, pred_class, pos_label=1)
    
    #sensitivity
    metrics['sensitivity'] = metrics['recall']
    metrics['fnr'] = 1 - metrics['sensitivity']
    
    # Specificity (Recall for class 0)
    metrics['specificity']  = recall_score(y_true, pred_class, pos_label=0)
    metrics['fpr']  = 1- metrics['specificity']
    
    #gmean
    metrics['gmean'] = np.sqrt(metrics['sensitivity']*metrics['specificity'])

    #balanced accuracy
    metrics['BA-score'] = balanced_accuracy_score(y_true, pred_class)

    #kappa
    metrics['kappa'] = cohen_kappa_score(y_true, pred_class)

    #probbaility metrics
    print(f"Log-Loss: {metrics['logloss']}")
    print(f"Brier score: {metrics['brier']}")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"PR-AUC: {metrics['pr_auc']}")
    
    #confusion matrix metrics
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1']}")
    print(f"Sensitivity: {metrics['sensitivity']}")
    print(f"Specificity: {metrics['specificity']}")
    print(f"False positive rate: {metrics['fpr']}")
    print(f"False negative rate: {metrics['fnr']}")
    print(f"G-Mean: {metrics['gmean']}")
    print(f"Balanced accuracy: {metrics['BA-score']}")
    print(f"Kappa: {metrics['kappa']}")

    return metrics


# # Multinomial Classification Metrics
# This function prints (and returns) all the multinomial classification metrics we discussed in the course. You would typically use it to evaluate your test set, or possibly a validation set.

# In[10]:


#y_true: actual class lables 
#pred_prob: predicted probabilities of each clas
#pred_class: predicted class label. 
# returns: prints (and returns) all multinomial classifcioant metrics: log-loss, AUC, accuracy, etc 
def compute_multinomial_class_metric(y_true, pred_prob, pred_class, lb):
    #store results in dict
    metrics = {}
    #logloss
    metrics['logloss'] = log_loss(y_true, pred_prob)
 
    #  brier
    metrics['brier'] =  multi_brier(y_true, pred_prob)
    
    #  aucroc
    metrics['auc_roc_ovr_macro'] = roc_auc_score(y_true, pred_prob, multi_class='ovr',  average='macro' )
    metrics['auc_roc_ovr_wght'] = roc_auc_score(y_true, pred_prob, multi_class='ovr',  average='weighted' )
    metrics['auc_roc_ovo_macro'] = roc_auc_score(y_true, pred_prob, multi_class='ovo',  average='macro' )
    metrics['auc_roc_ovo_wght'] = roc_auc_score(y_true, pred_prob, multi_class='ovo',  average='weighted' )
    
    #  aucprc
    metrics['pr_auc_ovr_macro'] = average_precision_score(y_true, pred_prob,   average='macro' )
    metrics['pr_auc_ovr_wght'] = average_precision_score(y_true, pred_prob,   average='weighted' )

   
    #accuracy   
    metrics['accuracy'] = accuracy_score(y_true, pred_class)

    # Precision, Recall, F1 Score, by class
    precision = precision_score(y_true, pred_class,  average=None)
    recall = recall_score(y_true, pred_class,  average=None)
    f1 = f1_score(y_true, pred_class, average=None)
    
    # create data frame to collect all values
    metrics['prec_recall_df']  = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }, index=lb.classes_)
    
    #gmean
    metrics['gmean'] = recall.prod()**(1.0/len(recall))

    #Macro Precision, Recall, F1 Score
    metrics['precision_mac']  = precision_score(y_true, pred_class,  average='macro')
    metrics['recall_mac'] = recall_score(y_true, pred_class,  average='macro')
    metrics['f1_mac'] = f1_score(y_true, pred_class, average='macro')

    #balanced
    metrics['BA-score'] = balanced_accuracy_score(y_true, pred_class)

    #kappa
    metrics['kappa'] = cohen_kappa_score(y_true, pred_class)

    #probbaility metrics
    print(f"Log-Loss: {metrics['logloss']}")
    print(f"Brier score: {metrics['brier']}")
    print(f"ROC-AUC, OVR macro: {metrics['auc_roc_ovr_macro']}")
    print(f"ROC-AUC, OVR weighted: {metrics['auc_roc_ovr_wght']}")
    print(f"ROC-AUC, OVO macro: {metrics['auc_roc_ovo_macro']}")
    print(f"ROC-AUC, OVO weighted: {metrics['auc_roc_ovo_wght']}")
    print(f"PR-AUC, OVR macro: {metrics['pr_auc_ovr_macro']}")
    print(f"PR-AUC, OVR weighted: {metrics['pr_auc_ovr_wght']}")

    
    #confusion matrix metrics
    print(f"Accuracy: {metrics['accuracy']}")
    # Print the DataFrame of precision, recall, f1
    print("Precision, Recall, F1 by class:")
    print(metrics['prec_recall_df'])
    print(f"G-Mean: {metrics['gmean']}")
    print(f"Macro Precision: {metrics['precision_mac']}")
    print(f"Macro Recall: {metrics['recall_mac']}")
    print(f"Macro F1: {metrics['f1_mac']}")
    print(f"Balanced accuracy: {metrics['BA-score']}")
    print(f"Kappa: {metrics['kappa']}")

    return metrics


# # G-mean
# 
# G-mean is the square root of the product of sensitivity and specificity. More generally, it is the geometric mean of the recall for each class (where recall represents the accuracy of each true class). Unfortunately, `sklearn` does not have a built-in function for calculating G-mean, so we define it here.
# 

# In[11]:


#y_true: actual class lables 
#y_pred: predicted class label.
#
# returns: g-mean 
def g_mean(y_true, y_pred):
    #first compute recall of each class
    recalls = recall_score(y_true, y_pred, average=None)
    #take geometric mean of all recalls
    geo_mean = recalls.prod()**(1/len(recalls))
    return geo_mean


# # Multinomial Brier score
# 
# The Brier score is similar to the MSE, but unfortunately, `sklearn` only provides built-in functionality for the binary case. Below, we compute the multinomial version of the Brier score.

# In[12]:


#y_true: actual class lables 
#pred_prob: predicted probabilities of each clas
#
# returns: brier score
def multi_brier(y_true, pred_prob):
    # One-hot encode the true labels to turn into 0/1
    encoder = OneHotEncoder(sparse_output=False )
    y_true_onehot = encoder.fit_transform(y_true.reshape(-1, 1))
    
    # Compute the squared differences
    squared_diff = (y_true_onehot - pred_prob) ** 2
    
    #sum across each observation
    sum_sq = np.sum(squared_diff,axis=1)

    #take mean
    return np.mean(sum_sq)


# # Tree Plots
# 
# Tree models offer a clear and intuitive visual representation of results. The code below generates a visualization of a decision tree.  
# 
# We have two main inputs parameter:
# 1.  **tree depth** determines how many levels of splits are displayed. If the depth is set too high, the tree may become overly complex and difficult to interpret. Choosing an appropriate depth helps ensure the visualization remains readable and informative.  
# 2. **feature_names** specifies the names of the features so we label the tree nodes correctly.
#  

# In[13]:


#plot tree
#tree_model: tree pipeline, last element model is the actual tree.
#depth: how many levels of tree to plto
#feature_names: names of features in model, so we lable tree correctly
#returns nothing, plots tree
def gen_tree_plot(tree_model,depth,feature_names,proportion=False):

    plt.figure(figsize=(30, 15)) # Resize figure
    tree.plot_tree(tree_model['model'],
                   max_depth = depth,
                   precision = 1,
                   filled = True,
                   impurity = False,
                   feature_names = feature_names,
                   fontsize=12,
                  rounded = True,
                  proportion=proportion)
    plt.show()

 


# # Importances
# 
# Many models compute a measure of feature importance to identify which features are most influential in making predictions. Below, we provide two functions to calculate and interpret these importance values.
# 

# ## Extract Importances
# 
# Below we extract the importances from the model. We present them in a sorted data frame

# In[14]:


#model: assumes pipeline model where 'preprocessor' should specify the variables included in model
#
#return: a dataframe lising all the importances, sorted by most important
def get_importances(model):

    #extract feature names from model
    feature_names = model['preprocessor'].get_feature_names_out()

    #actual importance values
    importances = model['model'].feature_importances_

    # Create a DataFrame for the feature importances
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort the DataFrame by importance
    return feature_importances.sort_values(by='Importance', ascending=False)


# ## Plot Importances
# 
# The below function plots the importances extracted from `get_importances`

# In[15]:


#model: assumes pipeline model where 'preprocessor' should specify the variables included in model
#
#return: nothing. plots  the importances, sorted by most important
def plot_importances(model):
    #call fundction to get ipmortances as df
    feature_importances = get_importances(model)

    # Plot the feature importances for better visualization
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()  # To display the most important feature at the top
    plt.show()

