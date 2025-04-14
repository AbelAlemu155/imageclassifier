"""Neural Network model."""

from .base import Module
from .optimizer import Optimizer

import numpy as np


def categorical_cross_entropy(pred, labels, epsilon=1e-10):
    """Cross entropy loss function.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).
    epsilon : float
        Small constant to add to the log term of cross entropy to help
        with numerical stability.

    Returns
    -------
    float
        Cross entropy loss.
    """
    if np.count_nonzero(pred < 0) > 0:
        print(f"**********************************************{pred}")
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))


def categorical_accuracy(pred, labels):
    """Accuracy statistic.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).

    Returns
    -------
    float
        Mean accuracy in this batch.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))


class Sequential:
    """Sequential neural network model.

    Parameters
    ----------
    modules : Module[]
        List of modules; used to grab trainable weights.
    loss : Module
        Final output activation and loss function.
    optimizer : Optimizer
        Optimization policy to use during training.
    """

    # set a default value for the loss???
    def __init__(self, modules, loss=None, optimizer=None):

        for module in modules:
            assert(isinstance(module, Module))
        # assert(isinstance(loss, Module))
        assert(isinstance(optimizer, Optimizer))

        self.modules = modules
        self.loss = loss

        self.params = []
        for module in modules:
            self.params += module.trainable_weights

        self.optimizer = optimizer
        self.optimizer.initialize(self.params)

    def forward(self, X, train=True):
        """Model forward pass.

        Parameters
        ----------
        X : np.array
            Input data

        Keyword Args
        ------------
        train : bool
            Indicates whether we are training or testing.

        Returns
        -------
        np.array
            Batch predictions; should have shape (batch, num_classes).
        """
        for module in self.modules:
            X = module.forward(X, train)
        X= self.loss.forward(X) 
    
        # Apply the final loss function after the last layer
        return X

    def backward(self, y):
        """Model backwards pass.

        Parameters
        ----------
        y : np.array
            True labels.
        """
        # Compute the loss gradient
        loss_grad = self.loss.backward(y)
        
        # Backpropagate the gradients through the layers
        for module in reversed(self.modules):
            
            loss_grad = module.backward(loss_grad)
        self.optimizer.apply_gradients(self.params)
        
    
    def train(self, dataset):
        """Fit model on dataset for a single epoch.

        Parameters
        ----------
        X : np.array
            Input images
        dataset : Dataset
            Training dataset with batches already split.

        Notes
        -----
        You may find tqdm, which creates progress bars, to be helpful:

        Returns
        -------
        (float, float)
            [0] Mean train loss during this epoch.
            [1] Mean train accuracy during this epoch.
        """
        """Fit model on dataset for a single epoch.

        Parameters
        ----------
        dataset : Dataset
            Training dataset with batches already split.

        Returns
        -------
        (float, float)
            [0] Mean train loss during this epoch.
            [1] Mean train accuracy during this epoch.
        """
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for X_batch, y_batch in dataset:
            # Forward pass
            predictions = self.forward(X_batch, train=True)
           
            # Compute the loss
            loss = categorical_cross_entropy(predictions, y_batch)
            
            total_loss += loss

            # Compute accuracy
            accuracy = categorical_accuracy(predictions, y_batch)
            total_accuracy += accuracy

            # Backward pass and gradient update
            self.backward(y_batch)
            # self.optimizer.apply_gradients(self.params)
            num_batches+=1

        # Compute mean loss and accuracy over the dataset
        mean_loss = total_loss / num_batches
        mean_accuracy = total_accuracy / num_batches
        return mean_loss, mean_accuracy

    def test(self, dataset):
        """Compute test/validation loss for dataset.

        Parameters
        ----------
        dataset : Dataset
            Validation dataset with batches already split.

        Returns
        -------
        (float, float)
            [0] Mean test loss.
            [1] Test accuracy.
        """


        total_loss = 0
        total_acc = 0
        count = 0
        
        for X, y in dataset:
            # Forward pass (with train=False)
            pred = self.forward(X, train=False)
            # Calculate loss and accuracy
            loss = categorical_cross_entropy(pred, y)
            acc = categorical_accuracy(pred, y)
            
            total_loss += loss
            total_acc += acc
            count += 1
            
        return total_loss / count, total_acc / count
