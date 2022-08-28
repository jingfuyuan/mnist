import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))


def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE
    prod = np.dot(theta, X.T)/temp_parameter
    max_prod = prod.max(axis=0)
    prod_shift = prod - max_prod
    prod_exp = np.exp(prod_shift)
    return prod_exp/prod_exp.sum(axis=0)


# this function is actually the same as the compute_probabilities function. Only the parameters name are different.
# def compute_probabilities_kernel(kernel_matrix, alpha, temperature):
#     """
#     Given the parameter alpha, calculate the probability of each data point X[i] being labeled as j
#     (j=0, 1, 2, ... k)
#     Args:
#         kernel_matrix: - (n, p) Numpy array (n data points with p features)
#         alpha: - (k, n) Numpy array, where jth row is the parameters for calculating the probability of label j
#         temperature: scalar value. Temperature parameter of the softmax function
#
#     Returns:
#         (k, n) numpy array, where each entry (i, j) is the probability of data point i being labeled j.
#     """
#     prod = np.dot(alpha, kernel_matrix) / temperature
#     prod_max = prod.max(axis=0)
#     prod_shift = prod - prod_max
#     prod_exp = np.exp(prod_shift)
#     return prod_exp / prod_exp.sum(axis=0)
def compute_probabilities_kernel(train_x, x_prime, alpha, temperature, kernel_function, **kwargs):
    axk = multiply_kernel(alpha, train_x, x_prime, kernel_function, **kwargs)
    prod = axk / temperature
    prod_shift = prod - prod.max(axis=0)
    prod_exp = np.exp(prod_shift)
    return prod_exp / prod_exp.sum(axis=0), axk


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    prob = compute_probabilities(X, theta, temp_parameter)
    loss = 0
    n = len(Y)
    for i, yi in enumerate(Y):
        loss -= np.log(prob[yi, i])/n
    reg = lambda_factor/2 * np.sum(theta**2)
    return loss + reg


# def compute_cost_function_kernel(kernel_matrix, labels, alpha, lambda_factor, temperature):
#     """
#
#     Args:
#         kernel_matrix: - (n, n) Numpy array
#         labels: - (n, ) Numpy array containing the labels for each data point (0-9)
#         alpha:  - (k, n) Numpy array. The ith row is the parameter for calculating the probability that a data point
#         is labeled as i
#         lambda_factor: - scalar value, the regularization parameter
#         temperature: - scalar value, the temperature parameter of softmax function
#
#     Returns:
#         c - the cost value (scalar)
#     """
#     prob = compute_probabilities_kernel(kernel_matrix, alpha, temperature)
#     loss = 0
#     n = len(labels)
#     for i, yi in enumerate(labels):
#         loss -= np.log(prob[yi, i]) / n
#     reg = lambda_factor / 2 * np.sum(alpha ** 2)
#     return loss + reg

def computer_cost_function_kernel(train_x, train_y, alpha, temperature, lambda_factor, kernel_function, **kwargs):
    prob, axk = compute_probabilities_kernel(train_x, train_x, alpha, temperature, kernel_function, **kwargs)
    loss = 0
    n = len(train_x)
    for i, yi in enumerate(train_y):
        loss -= np.log(prob[yi, i]) / n
    print("loss is ", loss)
    return loss + np.trace(np.dot(axk, alpha.T)) * lambda_factor / 2


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    k, n = len(theta), len(Y)
    # each column of y_mtx is one_hot encoded label. for example, y=2 is coded with [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y_mtx = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k, n)).toarray()
    prob = compute_probabilities(X, theta, temp_parameter)
    gradient = -1/temp_parameter/n * np.dot(y_mtx - prob, X) + lambda_factor * theta
    return theta - alpha * gradient


# def one_gd_iteration_kernel(kernel_matrix, labels, alpha, learning_rate, lambda_factor, temperature):
#     """
#     run one interation of the gradient descent algorithm
#     Args:
#         kernel_matrix: - (n, n) Numpy array
#         labels:  - (n, ) Numpy array containing the label of each data point (0-9)
#         alpha: - (k, n) Numpy array where ith row is the parameter for calculating the probability that
#         data point is labeled i
#         learning_rate: - scalar value. The learning rate parameter of gradient decent algorithm
#         lambda_factor: - scalar value. Regularization parameter
#         temperature:  - scalar value. Temperature parameter of softmax function
#
#     Returns:
#         updated_alpha: - (k, n) Numpy array
#     """
#     k, n = len(alpha), len(labels)
#     y_mtx = sparse.coo_matrix(([1] * n, (labels, range(n))), shape=(k, n)).toarray()
#     prob = compute_probabilities_kernel(kernel_matrix, alpha, temperature)
#     gradient = -1 / temperature / n * np.dot(y_mtx - prob, kernel_matrix) + lambda_factor * alpha
#     return alpha - learning_rate * gradient

def one_gd_iteration_kernel(train_x, train_y, alpha, learning_rate, lambda_factor,
                            temperature, kernel_function, **kwargs):
    k, n = len(alpha), len(train_x)
    y_matrix = sparse.coo_matrix(([1] * n, (train_y, range(n))), shape=(k, n)).toarray()
    prob, axk = compute_probabilities_kernel(train_x, train_x, alpha, temperature, kernel_function, **kwargs)
    prob_diff = y_matrix - prob
    # i = np.random.randint(0, n)
    batch_size = 500
    i = np.random.choice(n, batch_size, replace=False)
    k_i = kernel_function(train_x, train_x[i], **kwargs)
    gradient = -1 / temperature / batch_size * np.dot(prob_diff[:, i], k_i.T)
    gradient += lambda_factor * axk
    return alpha - learning_rate * gradient


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    return np.mod(train_y, 3), np.mod(test_y, 3)


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    pred_label = get_classification(X, theta, temp_parameter)
    y_mod3, pred_label_mod3 = np.mod(Y, 3), np.mod(pred_label, 3)
    return 1 - np.mean(y_mod3 == pred_label_mod3)


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression


# def softmax_regression_kernel(train_x, train_y, temperature, learning_rate, lambda_factor, k,
#                               kernel_function, num_iterations, **kwargs):
#     # calculating the kernel matrix
#     print("calculating kernel matrix...")
#     kernel_matrix = kernel_function(train_x, train_x, **kwargs)
#     n = len(train_x)
#     alpha = np.zeros((k, n))
#     cost_function_progression = []
#     for i in range(num_iterations):
#         print("Interation ", i)
#         cost_value = compute_cost_function_kernel(kernel_matrix, train_y, alpha, lambda_factor, temperature)
#         cost_function_progression.append(cost_value)
#         alpha = one_gd_iteration_kernel(kernel_matrix, train_y, alpha, learning_rate, lambda_factor, temperature)
#     return alpha, cost_function_progression

def softmax_regression_kernel(train_x, train_y, temperature, learning_rate, lambda_factor, k,
                              kernel_function, num_iterations, **kwargs):
    n = len(train_x)
    alpha = np.zeros((k, n))
    cost_function_progression = []
    for i in range(num_iterations):
        print(f"{i}/{num_iterations} iteration of softmax")
        cost_value = computer_cost_function_kernel(train_x, train_y, alpha, temperature,
                                                   lambda_factor, kernel_function, **kwargs)
        cost_function_progression.append(cost_value)
        alpha = one_gd_iteration_kernel(train_x, train_y, alpha, learning_rate,
                                        lambda_factor, temperature, kernel_function, **kwargs)
    return alpha, cost_function_progression


def get_classification_kernel(train_x, test_x, alpha, temperature, kernel_function, **kwargs):
    probabilities, _ = compute_probabilities_kernel(train_x, test_x, alpha, temperature, kernel_function, **kwargs)
    return np.argmax(probabilities, axis=0)


def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)


def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)


def multiply_kernel(matrix_a, kernel_m1, kernel_m2, kernel_function, **kwargs):
    """
    calculate the dot product of a matrix_a and a kernel matrix. The kernel matrix is the dot product of kernel_m1
    and kernel_m2.T
    Args:
        matrix_a: - (k, n) Numpy array
        kernel_m1:  - (n, p) Numpy array
        kernel_m2:  - (m, p) Numpy array
        kernel_function: - a callalbe kernel function
        **kwargs: other parameters for the kernel function

    Returns:
        (k, m) Numpy array
    """
    m = len(kernel_m2)
    block_size = 3000
    current_row = 0
    sub_arrays = []
    while current_row < m:
        sub_kernel_mtx = kernel_function(kernel_m1, kernel_m2[current_row:current_row+block_size], **kwargs)
        sub_arrays.append(np.dot(matrix_a, sub_kernel_mtx))
        current_row += block_size
    return np.hstack(sub_arrays)