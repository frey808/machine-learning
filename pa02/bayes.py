################################################################
# Machine Learning
# Programming Assignment - Bayesian Classifier
#
# bayes.py - functions for Bayesian classifier
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

import math
import numpy as np

from debug import *
from results_visualization import *

################################################################
# Cost matrices
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def uniform_cost_matrix( num_classes ):
    cost_matrix = np.ones((num_classes,num_classes))
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def bnrs_unequal_costs( num_classes ):
    # Rows: output class, Columns: Target (ground truth) class
    cost_matrix = np.matrix([[-0.2, 0.7, 0.7, 0.7],
                   [0.7, -0.15, 0.7, 0.7],
                   [0.7, 0.7, -0.05, 0.7],
                   [0.3, 0.3, 0.3, 0.3]])
    return cost_matrix

################################################################
# Bayesian parameters 
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def priors( split_data ):
    
    n_samples = sum(map(len, split_data))
    return [len(data_class)/n_samples for data_class in split_data]

def bayesian_parameters( CLASS_DICT, split_data, title='' ):
    # Compute class priors, means, and covariances matrices WITH their inverses (as pairs)
    class_priors = priors(split_data)
    class_mean_vectors = list( map( mean_vector, split_data ) )
    class_cov_matrices = list( map( covariances, split_data ) )

    # Show parameters if title passed
    if title != '':
        print('>>> ' + title)
        show_for_classes(CLASS_DICT, "[ Priors ]", class_priors )

        show_for_classes(CLASS_DICT, "[ Mean Vectors ]", class_mean_vectors)
        show_for_classes(CLASS_DICT, '[ Covariances and Inverse Covariances]', class_cov_matrices )
        print('')

    return (class_priors, class_mean_vectors, class_cov_matrices)


################################################################
# Gaussians (for class-conditional density estimates) 
################################################################

def mean_vector( data_matrix ):
    # Axis 0 is along columns (default)
    return np.mean( data_matrix, axis=0)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def covariances( data_matrix ):
    # HEADS-UP: The product of the matrix by its inverse may not be identical to the identity matrix
    #           due to finite precision. Can use np.allclose() to test for 'closeness'
    #           to ideal identity matrix (e.g., np.eye(2) for 2D identity matrix)
    d = data_matrix.shape[1]
    mv = mean_vector(data_matrix)
    cov_matrix = np.zeros((d, d))
    for sample in data_matrix:
        for x in range(d):
            for y in range(d):
                cov_matrix[x][y] += (sample[x] - mv[x]) * (sample[y] - mv[y])
    cov_matrix /= (len(data_matrix) - 1)

    # Returns a pair: ( covariance_matrix, inverse_covariance_matrix )
    return ( cov_matrix, np.linalg.inv(cov_matrix) )

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def mean_density( cov_matrix ):
    n_features = cov_matrix.shape[0]
    cov_determinant = np.linalg.det(cov_matrix)
    return (1 / np.pow(2 * np.pi, n_features/2)) * (1 / np.sqrt(cov_determinant))
 

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def sq_mhlnbs_dist( data_matrix, mean_vector, cov_inverse ):
    # Square of distance from the mean in *standard deviations* 
    # (e.g., a sqared mahalanobis distance of 9 implies a point is sqrt(9) = 3 standard
    # deviations from the mean.

    # Numpy 'broadcasting' insures that the mean vector is subtracted row-wise
    diff = data_matrix - mean_vector
    mhlnbs_dist = np.matrix([[d.T @ cov_inverse @ d] for d in diff])

    return mhlnbs_dist

def gaussian( mean_density, distances ):
    # NOTE: distances is a column vector of squared mahalanobis distances

    # Use numpy matrix op to apply exp to all elements of a vector
    scale_factor = np.exp( -0.5 * distances )

    # Returns Gaussian values as the value at the mean scaled by the distance
    return mean_density * scale_factor


################################################################
# Bayesian classification
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#     ** Where indicated
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def map_classifier( priors, mean_vectors, covariance_pairs ):
    # Unpack data once outside definition (to avoid re-computation)
    covariances =  np.array( [ cov_pair[0] for cov_pair in covariance_pairs ] )
    peak_scores = priors * np.array( [ mean_density(c) for c in covariances ] )

    inv_covariances =  np.array( [ cov_pair[1] for cov_pair in covariance_pairs ] )
    num_classes = len(priors)

    def classifier( data_matrix ):
        num_samples = data_matrix.shape[0]

        # Create arrays to hold distances and class scores
        distances = np.zeros( ( num_samples, num_classes ) )
        class_scores = np.zeros( ( num_samples, num_classes + 1 ) ) 

        #>>>>>>>>> EDIT THIS SECTION

        for i in range(num_classes):
            distances[:,i] = sq_mhlnbs_dist(data_matrix, mean_vectors[i], inv_covariances[i]).T
            class_scores[:,i] = gaussian(peak_scores[i], distances[:,i])

        class_scores[:,-1] = np.argmax(class_scores[:,:-1], axis=1)
        
        #>>>>>>>>>> END SECTION TO EDIT
        
        return class_scores

    return classifier


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#     ** Where indicated
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def bayes_classifier( cost_matrix, priors, mean_vectors, covariance_pairs ):
    # Unpack data once outside definition (to avoid re-computation)
    covariances =  np.array( [ cov_pair[0] for cov_pair in covariance_pairs ] )
    peak_scores = priors * np.array( [ mean_density(c) for c in covariances ] )

    inv_covariances =  np.array( [ cov_pair[1] for cov_pair in covariance_pairs ] )
    num_classes = len(priors)

    def classifier( data_matrix ):
        num_samples = data_matrix.shape[0]

        # Create arrays to hold distances and class scores
        distances = np.zeros( ( num_samples, num_classes ) )
        class_posteriors = np.zeros( ( num_samples, num_classes ) ) 
        class_costs_output = np.zeros( ( num_samples, num_classes + 1) ) 

        #>>>>>>>>> EDIT THIS SECTION
        
        for i in range(num_classes):
            distances[:,i] = sq_mhlnbs_dist(data_matrix, mean_vectors[i], inv_covariances[i]).T
            class_posteriors[:,i] = gaussian(peak_scores[i], distances[:,i])
        
        for pred in range(num_classes):
            for gt in range(num_classes):
                class_costs_output[:,pred] += cost_matrix[pred,gt] * class_posteriors[:,gt]

        class_costs_output[:,-1] = np.argmin(class_costs_output[:,:-1], axis=1)
        
        #>>>>>>>>>> END SECTION TO EDIT

        return class_costs_output

    return classifier
