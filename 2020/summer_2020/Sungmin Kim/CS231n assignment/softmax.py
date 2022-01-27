from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
    denominator = 0

    for i in range(len(X)):
        score = np.dot(X[i], W)
        score -= np.max(score)
        correct_class_score = score[y[i]]

        denominator = np.sum(np.exp(score))
			
        loss += -np.log(np.exp(correct_class_score) / denominator)
		
        for j in range(len(score)):
            if j != y[i]:
                dW[:, j] += (np.exp(score[j]) / denominator) * X[i]
        dW[:, y[i]] += ((np.exp(correct_class_score) / denominator) - 1) * X[i]
		
    loss /= X.shape[0]
    dW /= X.shape[0]

    loss += reg * np.sum(W ** 2)
    dW += reg * (2 * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score = np.dot(X, W)
    score -= np.max(score)

    denominator = np.sum(np.exp(score), axis=1)
    softmax = np.exp(score) / denominator.reshape(X.shape[0], -1)
    correct_class_score = np.exp(score[np.arange(X.shape[0]), y])
	
    loss += np.sum(-np.log(correct_class_score/denominator))
	
    softmax[np.arange(X.shape[0]), y] -= 1
    dW = np.dot(X.T, softmax)
	
    loss /= X.shape[0]
    dW /= X.shape[0]
	
    loss += reg * np.sum(W ** 2)
    dW += reg * (2 * W)
	
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
