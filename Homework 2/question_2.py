# -*- coding: utf-8 -*-
"""
Author: Irem Ecem Yelkanat
Written with Python v3.8.2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linear_regression(features, labels):

    # Find beta coefficient(s)
    beta = np.dot(np.dot(np.linalg.inv(np.dot(features.T, features)), features.T), labels)

    # Predict samples
    predictions = np.dot(features, beta)

    # Sort matrices according to sqft living
    permutation = np.argsort(features[:,1])
    sorted_features = (features[:,1])[permutation]
    sorted_predictions = predictions[permutation]

    # Calculate loss (MSE)
    prediction_difference = labels - predictions
    prediction_difference_squared = np.square(prediction_difference)
    error_sum = prediction_difference_squared.sum()
    loss = error_sum / len(labels)

    return beta, sorted_features, sorted_predictions, loss['price']


def plot_regression(features, labels, sorted_features, sorted_predictions, label_addition=""):

    plt.plot(sorted_features, sorted_predictions, label="predicted")
    plt.scatter(np.array(features[:,1]), np.array(labels), s=2, color="r", label="actual")
    plt.legend(loc="upper left")
    plt.title('sqftliving vs. price' + label_addition)
    plt.xlabel('sqftliving')
    plt.ylabel('price')
    plt.show()



if __name__ == "__main__":
    # Load datasets
    train_X = pd.read_csv('question-2-features.csv')
    train_Y = pd.read_csv('question-2-labels.csv')

    # Process data
    labels = train_Y
    x_O = [[1]] * len(labels)
    features = np.append(x_O, train_X, axis=1)

    # Find and print rank
    xx_t = np.dot(features.T, features)
    print("Rank of mutiplication of X and X transpose (XX^T) is:", np.linalg.matrix_rank(xx_t))

    # Train only using sqft_living
    print("\n\n----Training only using sqft_living feature----")

    # Get sqftliving feature
    features_0 = features[:,:2]

    # Train data
    beta, sorted_features, sorted_predictions, loss = linear_regression(features_0, labels)

    # Print coefficients in beta
    print("\nCoefficients are:")
    for i in range(beta.shape[0]):
        pass
        print("b_" + str(i) + " is "+ str(beta[i][0]))
    # Report error
    print("Calculated Mean Squared Error(MSE) is: " + str(loss))

    # Plot price vs. ”sqftliving”
    plot_regression(features_0, labels, sorted_features, sorted_predictions)

    

    # Train only using sqft_living along with its square
    print("\n\n----Training using sqft_living feature and its square, polynomial regression----")

    # Get sqftliving feature
    features_1 = features[:,:2] 

    # Generate matrix containing sqft_living along with its square
    dim = (features_1.shape)[0]
    sqftliving_square = (np.square(features_1)[:,1]).reshape(((features_1.shape)[0], 1))
    features_square = np.append(features_1, sqftliving_square, axis=1)

    # Train data
    beta, sorted_features, sorted_predictions, loss = linear_regression(features_square, labels)

    # Print coefficients in beta
    print("\nCoefficients are:")
    for i in range(beta.shape[0]):
        print("b_" + str(i) + " is "+ str(beta[i][0]))
        
    # Report error
    print("Calculated Mean Squared Error(MSE) is: " + str(loss))

    # Plot price vs. ”sqftliving”
    plot_regression(features, labels, sorted_features, sorted_predictions, " polynomial included")
