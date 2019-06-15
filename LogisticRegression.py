import numpy as np

class LogisticRegression(object):
    """
    Logistic Regression classifier.
    
    Hyperarameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    epochs: int
        Number of iterations
    random_state : Random generator seed for randomizing weight
    
    Parameters
    -----------
    w_ : 1d array
        Weights after fitting
    b_ : scalar
        bias
    costs_ : list
        costs history after every epoch
    accuracy_ : list
        Accuracy history after every epoch
    """
    
    def __init__(self,eta=0.01,epochs=50,random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = 1
        
    def initialize_weights(self,input_shape):
        """
        Initialize weights.
        
        Input
        ---------
        input_shape : 1d array
            No. of features
        """
        np.random.seed(self.random_state)
        self.w_ = np.random.randn(input_shape)
        self.b_ = 0
        self.costs_ = []
        self.accuracy_ = []
        
        return
        
    def sigmoid(self,z):
        out = 1./(1+np.exp(-z))
        return out
    
    def predict(self,X):
        """
        Predict the output
        
        Input
        -----------
        X: 2d array
            Input array of shape (num_features,num_samples)
        
        Returns
        -----------
        y_pred: 1d array
            Returns 1d array of predicted values of shape (1,num_samples)
        """
        y_pred = self.sigmoid(np.dot(self.w_.T,X) + b)
        
        return y_pred
    
    def compute_cost(self,y,y_pred):
        """
        Computes cost (log likelihood) and accuracy.
        Appends it to costs_ and accuracy_
        
        Input
        -------
        y_pred: 1d array
            Predicted output of shape (1,num_samples)
        y: 1d array
            True output of shape (1,num_samples)
        """
        
        cost = -[y*np.log(y_pred) + (1-y)*np.log(1-y_pred)]
        self.costs_.append(cost)
        
        accuracy = np.sum(y_pred==y)/y.shape[1]
        self.accuracy_.append(accuracy)
        
        return
    
    def update_parameters(self,X,y,y_pred):
        """
        Update parameters
        
        """
        
        gradient_w = np.dot(X.T,(y_pred-y))/y.shape[1]
        gradient_b = np.sum(y_pred-y)
        
        self.w_ -= eta*gradient_w
        self.b_ -= eta*gradient_b
        
        return
    
    def fit(self,X,y):
        """
        Fit X to y. Learn parameters w_ and b_
        
        Input
        --------
        X: 2d array
            Input array of shape (num_features,num_samples)
        y: 1d array
            True output labels of shape (1,num_samples)
        
        Returns
        --------
        self object
        
        """
        
        for _ in range(self.epochs):
            #Initialize weights
            input_shape = X.shape[0]
            self.initialize_weights(input_shape)

            #Predict output on this weights
            y_pred = self.predict(X)

            #Compute cost
            self.compute_cost(y,y_pred)

            #Update parameters
            self.update_parameters(X,y,y_pred)
        
        