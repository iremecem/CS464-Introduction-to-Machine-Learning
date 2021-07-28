# -*- coding: utf-8 -*-
"""
Author: Irem Ecem Yelkanat
Written with Python v3.8.2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Find top Principal Components
def find_top_PCs(X):

    # Mean center the data and compute eigen values and eigen vectors
    mean = X.mean(axis=0)
    covariance_matrix = np.cov((X - mean), rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # Sort eigen values and eigen vectors parallel according to eigen values
    permutation = eigen_values.argsort()[::-1]
    eigen_values_sorted = (eigen_values[permutation])
    eigen_vectors_sorted = (eigen_vectors[:,permutation])

    # Find proportion of expla,ned variance for each component
    PVEs = []
    for i in range(len(eigen_values)):
        PVEs.append(eigen_values[i] / np.sum(eigen_values))

    return (eigen_values_sorted, eigen_vectors_sorted, PVEs)

if __name__ == "__main__":
    # Read data
    digits_data = pd.read_csv('digits.csv')

    # Split data to features and labels
    features = digits_data.drop(labels='label', axis=1).values
    labels = digits_data['label'].values

    # Calculate eigenvalues, eigenvectors and PVEs
    eigen_values, eigen_vectors, PVEs = find_top_PCs(features)

    # Report the proportion of variance explained (PVE) for each of the principal components.
    for index in range(10):
        print("For Principal Component #" + str(index + 1) + ", PVE is", (PVEs[index]).real)

    # Reshape each of the principal component to a 28x28 matrix and show them.
    first_10_pc_fig = plt.figure()
    first_10_pc_fig.suptitle("First 10 Principal Components")
    for index in range(10):
        pc = (eigen_vectors[:,index].reshape(28,28)).real # get real part of the vector
        first_10_pc_fig.add_subplot(2, 5, index + 1, title=("PC #" + str(index + 1)))
        plt.imshow(pc, cmap="gray") # show the image as black & white
        plt.axis('off') # turn off axis ticks
    plt.show(block=True)

    '''
    Obtain first k principal components
    and report PVE for k ∈ {8,16,32,64,128,256}.
    Plot k vs. PVE.
    '''
    ks_for_plot = [8, 16, 32, 64, 128, 256]
    PVEs_sum = []
    PVEs_individual = []
    for k in (2**p for p in range(0, 6)):
        PVEs_individual.append((PVEs[8 * k]).real) # Store individual PVE value
        PVEs_sum.append(np.sum(PVEs[:(8 * k)]).real) # Store sums of PVE values


    # Plot k vs. PVE (individual)
    plt.plot(ks_for_plot, PVEs_individual)
    for k, pve in zip(ks_for_plot, PVEs_individual): 
        plt.text(k, pve, str(round(pve, 4)))
    plt.title('k vs. PVE (individual)')
    plt.xlabel('k')
    plt.ylabel('PVE')
    plt.show(block=True)

    # Plot k vs. PVE (sum)
    plt.plot(ks_for_plot, PVEs_sum)
    for k, pve in zip(ks_for_plot, PVEs_sum): 
        plt.text(k, pve, str(round(pve, 2)))
    plt.title('k vs. PVE (sum)')
    plt.xlabel('k')
    plt.ylabel('PVE')
    plt.show(block=True)

    '''
    Use first k principal components to analyze and 
    reconstruct the first image in the dataset 
    where k ∈ 1, 3, 5, 10, 50, 100, 200, 300.
    '''
    k_values = [1, 3, 5, 10, 50, 100, 200, 300]
    images = []
    mean_data = features.mean(axis=0) # Find the mean of the data
    centered_data = features - mean_data # Mean center the data

    for k in k_values:
        selected_vectors = eigen_vectors[:,:k] # Select first k vectors
        PCA_scores = centered_data.dot(selected_vectors) # Compute PCA scores
        X_hat = PCA_scores.dot(selected_vectors.T) + mean_data # Reconstruct data
        first_digit = X_hat[0] # Fet first digit
        first_digit = (first_digit.reshape(28, 28)).real # Get the real part and reshape to 28x28
        images.append(first_digit)

    num_images = len(images)

    # Show images for k vs. Reconstructed Image
    reconstructed_images = plt.figure()
    reconstructed_images.suptitle('k vs. Reconstructed Image')
    reconstructed_images.subplots_adjust(hspace=.5)
    for i in range(num_images):
        reconstructed_images.add_subplot(3, 3, i + 1, title=("k=" + str(k_values[i])))
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    reconstructed_images.add_subplot(3, 3, num_images + 1, title=("Original"))
    plt.imshow(features[0].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.show(block=True)

    # Show images for k vs. Reconstructed Image (rescaled)
    reconstructed_images_rescaled = plt.figure()
    reconstructed_images_rescaled.suptitle('k vs. Reconstructed Image (rescaled)')
    reconstructed_images_rescaled.subplots_adjust(hspace=.5)
    for i in range(num_images):
        reconstructed_images_rescaled.add_subplot(3, 3, i + 1, title=("k=" + str(k_values[i])))
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
    reconstructed_images_rescaled.add_subplot(3, 3, num_images + 1, title=("Original"))
    plt.imshow(features[0].reshape(28,28), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show(block=True)

