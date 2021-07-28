# -*- coding: utf-8 -*-
"""
Author: Irem Ecem Yelkanat
Written with Python v3.8.2
"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import random

def batchGradientAscent(learning_rate, features, labels, epoch):

    # Initialize W
    W = np.zeros(30)

    for i in range(epoch):

        # Calculate y_hat for y=1
        z = np.exp(np.add(W[0], np.dot(features, W[1:])))
        Y_1_W_X = np.divide(z, np.add(z, 1))

        Y_1_W_X = Y_1_W_X.reshape(features.shape[0], 1)
        difference = labels - Y_1_W_X

        # Update weights
        W[0] += learning_rate * np.sum(difference)
        W[1:] += (learning_rate * (np.dot(np.transpose(features), difference))).reshape(29,)

    return W



def miniBatchGradientAscent(learning_rate, features, labels, epoch, batch_size):

    num_samples = features.shape[0]
    np.random.seed(42)
    W = np.random.normal(0, 0.01, 30)

    for i in range(epoch):

        for j in range(0, num_samples, batch_size):

            # Get samples from data
            train_X_samples = features[j:(j + batch_size)]
            train_Y_samples = labels[j:(j + batch_size)]

            # Calculate y_hat for y=1
            z = np.exp(np.add(W[0], np.dot(train_X_samples, W[1:])))
            Y_1_W_X = np.divide(z, np.add(z, 1))
            
            Y_1_W_X = Y_1_W_X.reshape(train_X_samples.shape[0], 1)
            difference = train_Y_samples - Y_1_W_X

            W[0] += learning_rate * np.sum(difference)
            W[1:] += (learning_rate * (np.dot(np.transpose(train_X_samples), difference))).reshape(29,)

    return W


def stochasticGradientAscent(learning_rate, features, labels, epoch):
    
    num_samples = features.shape[0]
    np.random.seed(42)
    W = np.random.normal(0, 0.01, 30)

    for i in range(epoch):

        for j in range(0, num_samples):

            # Get sample from data
            train_X_sample = features[j]
            train_Y_sample = labels[j]

            # Calculate y_hat for y=1
            z = np.exp(np.add(W[0], np.dot(train_X_sample, W[1:])))
            Y_1_W_X = np.divide(z, np.add(z, 1))

            # Update weights
            difference = train_Y_sample - Y_1_W_X
            W[0] += learning_rate * np.sum(difference)
            W[1:] += (learning_rate * (np.dot((train_X_sample.reshape(29,1)), difference)).reshape(29,))

    return W



def analyze_model(features, labels, W, title=""):
    
    labels = np.transpose(labels)

    # Calculate predictions
    res = W[0] + features.dot(W[1:])
    res = res > 0
    predictions = np.where(res == True, 1, res)

    # Calculate the prediction situation
    prediction_situation = predictions == labels

    # Find the number of false, true and total number of predictions
    num_false_predictions = np.count_nonzero(prediction_situation == False)
    num_true_predictions = np.count_nonzero(prediction_situation == True)
    num_predictions = len(prediction_situation)

    # Calculate true spam, false spam, true normal, false normal values
    fraud_labels = labels == 1
    true_fraud_predictions = fraud_labels & prediction_situation

    num_true_fraud_predictions = np.count_nonzero(true_fraud_predictions == True)
    num_true_normal_predictions = num_true_predictions - num_true_fraud_predictions

    num_false_fraud_predictions = np.count_nonzero(predictions == 1) - num_true_fraud_predictions
    num_false_normal_predictions = num_false_predictions - num_false_fraud_predictions

    # Calculate and report performance metrics
    accuracy, precision, recall, negative_predictive_value, false_positive_rate, false_discovery_rate, f1_score, f2_score = calculateMetrics(num_true_fraud_predictions, num_false_fraud_predictions, num_true_normal_predictions, num_false_normal_predictions)
    print("Accuracy is:", accuracy)
    print("Precision is:", precision)
    print("Recall is:", recall)
    print("Negative Predictive Value is:", negative_predictive_value)
    print("False Positive Rate is:", false_positive_rate)
    print("False Discovery Rate is:", false_discovery_rate)
    print("F1 Score is:", f1_score)
    print("F2 Score is:", f2_score)

    # Plot confusion matrix
    confusion_matrix = [[num_true_fraud_predictions, num_false_normal_predictions],
                    [num_false_fraud_predictions, num_true_normal_predictions]]

    df_confusion_matrix = pd.DataFrame(confusion_matrix, ['Fraud', 'Normal'], ['Fraud', 'Normal'])
    sn.set(font_scale=1.4)
    sn.heatmap(df_confusion_matrix, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    title = "Confusion Matrix for " + title
    plt.title(title)
    plt.show()


def calculateMetrics(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    negative_predictive_value = TN / (TN + FN)
    false_positive_rate = FP / (FP + TN)
    false_discovery_rate = FP / (TP + FP)
    f1_score = (2 * precision * recall) / (precision + recall)
    f2_score = (5 * precision * recall) / ((4 * precision) + recall)
    return accuracy, precision, recall, negative_predictive_value, false_positive_rate, false_discovery_rate, f1_score, f2_score



if __name__ == "__main__":

    # Load datasets
    train_X = pd.read_csv('question-3-features-train.csv')
    train_Y = pd.read_csv('question-3-labels-train.csv')
    test_X = pd.read_csv('question-3-features-test.csv')
    test_Y = pd.read_csv('question-3-labels-test.csv')

    # Normalize Amount feature
    train_X['Amount'] = train_X['Amount'] / max(train_X['Amount'])
    test_X['Amount'] = test_X['Amount'] / max(test_X['Amount'])

    # Transform dataframes to numpy arrays
    train_X = train_X.values
    train_Y = train_Y.values
    test_X = test_X.values
    test_Y = test_Y.values

    # ----- FULL BATCH -----
    print("Running gradient ascent for full batch...")
    w_full = batchGradientAscent(1e-3, train_X, train_Y, 1000)
    print("Gradient ascent for full batch is done")

    print("\n---Analysis of Full-Batch Gradient Ascent---")
    analyze_model(test_X, test_Y, w_full, "Full-Batch")

    # ----- MINI BATCH -----
    print("\nRunning gradient ascent for mini batch...")
    w_mini = miniBatchGradientAscent(1e-3, train_X, train_Y, 1000, 100)
    print("Gradient ascent for mini batch is done")

    print("\n---Analysis of Mini-Batch Gradient Ascent---")
    analyze_model(test_X, test_Y, w_mini, "Mini-Batch")

    # ----- STOCHASTIC -----
    print("\nRunning gradient ascent for stochastic...")
    w_stochastic = stochasticGradientAscent(1e-3, train_X, train_Y, 1000)
    print("Gradient ascent for stochastic is done")

    print("\n---Analysis of Stochastic Gradient Ascent---")
    analyze_model(test_X, test_Y, w_stochastic, "Stochastic")