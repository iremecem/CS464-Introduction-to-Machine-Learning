# -*- coding: utf-8 -*-
"""
CS464 Introduction to Machine Learning
Homework 1
Author: Irem Ecem Yelkanat
21702624
"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import random
import copy
import math

# Load datasets
train_X = pd.read_csv('x_train.csv', header=None)
train_Y = pd.read_csv('y_train.csv', header=None)
test_X = pd.read_csv('x_test.csv', header=None)
test_Y = pd.read_csv('y_test.csv', header=None)

# Split train features according to classes
train_X_normal = train_X.loc[(train_Y[0] == 0)]
train_X_spam = train_X.loc[(train_Y[0] == 1)]

# Compute number of samples in training set as normal and spam
N = len(train_X)
N_normal = len(train_X_normal)
N_spam = N - N_normal

print("Total number of emails in the training set", N)
print("Number of spam emails in the training set", N_spam)
print("Number of normal emails in the training set", N_normal)
print("The percentage of the spam emails in the dataset", (N_spam / N))

# Calculate prior probabilities
pi_normal = N_normal / N
pi_spam = 1 - pi_normal

def calculateEstimatesMultinomialNB(train_X_normal, train_X_spam, alpha=0):

    # Get vocabulary size
    vocabularySize = train_X_normal.shape[1]

    # Calculate number of occurances for each word in training set for normal emails
    T_j_normal = train_X_normal.sum(axis = 0)
    T_j_normal = T_j_normal.to_numpy()

    # Calculate estimators for occurance for each word for normal emails
    num_words_in_normal = np.sum(T_j_normal)
    theta_j_normal = (T_j_normal + alpha) / (num_words_in_normal + (alpha * vocabularySize)) 

    # Calculate estimators for occurance for each word for spam emails
    T_j_spam = train_X_spam.sum(axis = 0)
    T_j_spam = T_j_spam.to_numpy()

    # Calculate estimators for occurance for each word for spam emails
    num_words_in_spam = np.sum(T_j_spam)
    theta_j_spam = (T_j_spam + alpha) / (num_words_in_spam + (alpha * vocabularySize))

    return (theta_j_normal, theta_j_spam)

def calculateEstimatesBernoulliNB(train_X_normal, train_X_spam, N_normal, N_spam):

    train_X_normal_copy = copy.deepcopy(train_X_normal)
    train_X_spam_copy = copy.deepcopy(train_X_spam)

    train_X_normal_copy[train_X_normal_copy > 0] = 1
    train_X_spam_copy[train_X_spam_copy > 0] = 1

    # Calculate number of emails for each word occured in training set for normal emails
    S_j_normal = train_X_normal_copy.sum(axis = 0)
    S_j_normal = S_j_normal.to_numpy()

    # Calculate estimators for occurance for each word for normal emails
    theta_j_normal = S_j_normal / N_normal

    # Calculate number of emails for each word occured in training set for spam emails
    S_j_spam = train_X_spam_copy.sum(axis = 0)
    S_j_spam = S_j_spam.to_numpy()

    # Calculate estimators for occurance for each word for spam emails
    theta_j_spam = S_j_spam / N_spam

    return (theta_j_normal, theta_j_spam)

def predictMultinomialNB(pi_normal, pi_spam, theta_j_normal, theta_j_spam, test_X):
    
    # Get the log of prior probability of normal emails
    pi_normal_log = np.log(pi_normal)

    # Get the log of estimators of normal emails
    with np.errstate(divide='ignore'):
        pi_normal_conditionals = np.log(theta_j_normal)

    # Calculate second part of the probability for normal emails
    pi_normal_conditionals_multiplied = test_X * pi_normal_conditionals

    # Calculate the probabilites for normal emails
    pi_normal_predictions = pi_normal_log + pi_normal_conditionals_multiplied.sum(axis=1)


    # Get the log of prior probability of spam emails
    pi_spam_log = np.log(pi_spam)

    # Get the log of estimators of spam emails
    with np.errstate(divide='ignore'):
        pi_spam_conditionals = np.log(theta_j_spam)

    # Calculate second part of the probability for spam emails
    pi_spam_conditionals_multiplied = test_X * pi_spam_conditionals

    # Calculate the probabilites for spam emails
    pi_spam_predictions = pi_spam_log + pi_spam_conditionals_multiplied.sum(axis=1)

    # Compare predictions
    prediction_comparison = pi_spam_predictions > pi_normal_predictions

    # Label the predictions accordingly
    predictions = np.where(prediction_comparison == True, 1, prediction_comparison)

    return predictions

def predictBernoulliNB(pi_normal, pi_spam, theta_j_normal, theta_j_spam, test_X):

    test_X_copy = copy.deepcopy(test_X)

    test_X_copy[test_X_copy > 0] = 1

    # Get the log of prior probability of normal emails
    pi_normal_log = np.log(pi_normal)

    # Store the estimators of normal emails
    pi_normal_conditionals = theta_j_normal

    # Calculate second part of the probability for normal emails
    pi_normal_conditionals_multiplied = (test_X_copy * pi_normal_conditionals) + ((1 - test_X_copy) * (1 - pi_normal_conditionals))

    # Calculate the log of the probabilities for normal emails
    with np.errstate(divide='ignore'):
        pi_normal_conditionals_multiplied_log = np.log(pi_normal_conditionals_multiplied)

    # Calculate the predictions for normal emails
    pi_normal_predictions = pi_normal_log + pi_normal_conditionals_multiplied_log.sum(axis=1)


    # Get the log of prior probability of spam emails
    pi_spam_log = np.log(pi_spam)

    # Store the estimators of spam emails
    pi_spam_conditionals = theta_j_spam

    # Calculate second part of the probability for spam emails
    pi_spam_conditionals_multiplied = (test_X_copy * pi_spam_conditionals) + ((1 - test_X_copy) * (1 - pi_spam_conditionals))

    # Calculate the log of the probabilities for normal emails
    with np.errstate(divide='ignore'):
        pi_spam_conditionals_multiplied_log = np.log(pi_spam_conditionals_multiplied)

    pi_spam_predictions = pi_spam_log + pi_spam_conditionals_multiplied_log.sum(axis=1)

    # Compare predictions
    prediction_comparison = pi_spam_predictions > pi_normal_predictions

    # Label the predictions accordingly
    predictions = np.where(prediction_comparison == True, 1, prediction_comparison)

    return predictions

def analyseModel(test_Y, predictions, modelType):

    # Get the test labels
    test_labels = test_Y[0].to_numpy()

    # Calculate the prediction situation
    prediction_situation = predictions == test_labels

    # Find the number of false, true and total number of predictions
    num_false_predictions = np.count_nonzero(prediction_situation == False)
    num_true_predictions = np.count_nonzero(prediction_situation == True)
    num_predictions = len(prediction_situation)

    # Calculate accuracy
    accuracy = num_true_predictions / num_predictions

    # Print metrics
    print("Total number of predictions:", num_predictions)
    print("Number of true predictions:", num_true_predictions)
    print("Number of wrong predictions:", num_false_predictions)
    print("Accuracy is:", accuracy)

    # Calculate true spam, false spam, true normal, false normal values
    spam_labels = test_labels == 1
    true_spam_predictions = spam_labels & prediction_situation

    num_true_spam_predictions = np.count_nonzero(true_spam_predictions == True)
    num_true_normal_predictions = num_true_predictions - num_true_spam_predictions

    num_false_spam_predictions = np.count_nonzero(predictions == 1) - num_true_spam_predictions
    num_false_normal_predictions = num_false_predictions - num_false_spam_predictions

    # Plot confusion matrix
    confusion_matrix = [[num_true_spam_predictions, num_false_normal_predictions],
                    [num_false_spam_predictions, num_true_normal_predictions]]

    df_confusion_matrix = pd.DataFrame(confusion_matrix, ['Spam', 'Normal'], ['Spam', 'Normal'])
    sn.set(font_scale=1.4)
    sn.heatmap(df_confusion_matrix, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    title = "Confusion Matrix for Bernoulli Naive Bayes Model" if modelType == "bnb" else "Confusion Matrix for Multinomial Naive \nBayes Model without smoothing" if modelType == "mnbwos" else "Confusion Matrix for Multinomial Naive \nBayes Model with smoothing"
    plt.title(title)
    plt.show()

# Multinomial Naive Bayes Without Smoothing
print("\n-----Running Multinomial Naive Bayes Model Without Smoothing-----\n")

# Calculate the estimates
theta_j_normal_mnbwos, theta_j_spam_mnbwos = calculateEstimatesMultinomialNB(train_X_normal, train_X_spam)
predictions_mnbwos = predictMultinomialNB(pi_normal, pi_spam, theta_j_normal_mnbwos, theta_j_spam_mnbwos, test_X)
analyseModel(test_Y, predictions_mnbwos, "mnbwos")


# Multinomial Naive Bayes With Smoothing
print("\n-----Running Multinomial Naive Bayes Model With Smoothing-----\n")

# Calculate the estimates
theta_j_normal_mnbws, theta_j_spam_mnbws = calculateEstimatesMultinomialNB(train_X_normal, train_X_spam, 1)
predictions_mnbws = predictMultinomialNB(pi_normal, pi_spam, theta_j_normal_mnbws, theta_j_spam_mnbws, test_X)
analyseModel(test_Y, predictions_mnbws, "mnbws")


# Bernoulli Naive Bayes
print("\n-----Running Bernoulli Naive Bayes Model-----\n")

# Calculate the estimates
theta_j_normal_bnb, theta_j_spam_bnb = calculateEstimatesBernoulliNB(train_X_normal, train_X_spam, N_normal, N_spam)
predictions_bnb = predictBernoulliNB(pi_normal, pi_spam, theta_j_normal_bnb, theta_j_spam_bnb, test_X)
analyseModel(test_Y, predictions_bnb, "bnb")

