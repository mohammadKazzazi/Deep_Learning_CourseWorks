import torch

class SGD:
    def __init__(self, params, learning_rate):
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.
        
        Args:
            params: List of model parameters to be updated.
            learning_rate: The learning rate for updating the parameters.
        """
        self.params = list(params)  # Model parameters to be updated
        self.learning_rate = learning_rate

    def step(self):
        """
        Perform a parameter update using Stochastic Gradient Descent (SGD).
        """
        with torch.no_grad():  # Disable gradient tracking during the update
            for param in self.params:
                if param.grad is not None:
                    param.sub_(self.learning_rate * param.grad)  # Update parameter in place

    def zero_grad(self):
        """
        Zero the gradients for all parameters.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad = None  # Clear the gradient for each parameter