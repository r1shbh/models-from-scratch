import numpy as np


class Adaline(object):
    """
    Adaptive Linear Neuron Classifier
    
    Hyperparameters
    -------------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Number of iterations to pass through training dataset
    random_state: int
        Randomized seed
        
    Parameters
    --------------
    w_: 1d-array
        Weight vector of size [n_features, 1]
    b_: scalar
        Bias weight
    costs_: list
        cost function after every iteration
    """
    
    def __init__(self, eta=0.1, n_iter=50, random_state=2):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Trains the weights of the model based on input vectors and target values.
        Input:
            X: Array [n_samples, n_features]
                Training vectors
            y: Array [n_samples, 1]
                Target values
                
        Output:
            self object
        """
        #Setting seed
        np.random.seed(self.random_state)
        
        #Charecterizing input arrays
        y = y.reshape(y.shape[0], 1)
        n_samples = y.shape[0]
        assert(n_samples==X.shape[0])
        n_features = X.shape[1]
        
        #Initializing weights/parameters
        self.w_ = np.random.randn(n_features, 1)*0.01
        self.b_ = np.random.randn()*0.01
        self.costs_ = []
        
        #Fitting
        for _ in range(self.n_iter):
            # Predict activation
            y_act = self.activation(X)
            
            #Updating cost
            cost = 0.5*np.sum((y-y_act)**2)
            self.costs_.append(cost)
            
            #Updating weights
            self.w_ -= self.eta*np.dot(X.T, y_act - y)
            self.b_ -= self.eta*np.sum(y_act - y)
        return self
    
    def activation(self, X):
        """
        Activation function (linear)
        Input
        ------------
        X: 2d-array
            Input array vector of size [n_samples, n_features]
        Output:
        y_act: 1d-array
            Activated output of size [n_samples, 1]
        """
        
        y_act = np.dot(X, self.w_) + self.b_
        return y_act.reshape(y_act.shape[0], 1)
    
    def predict(self, X):
        """
        Input:
            X: Array of size [n_samples, n_features]
                Input array vector
        Output:
            y_predict: Size [n_samples, 1]
                Predicted target values
        """ 
        y_predict = np.where(self.activation(X)>=0.0, 1.0, -1.0)
        return y_predict
    
    
    
class Perceptron(object):
    def __init__(self, eta = 0.1, n_iter = 50, random_state = 2):
        self.eta = 0.1
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Trains the weights of the model based on input vectors and target values.
        Input:
            X: Array [n_samples, n_features]
                Training vectors
            y: Array [n_samples, 1]
                Target values
                
        Output:
            self object
        """
        
        np.random.seed(self.random_state)
        
        # Charecterizing the input arrays
        n_samples = y.shape[0]
        assert(n_samples==X.shape[0])
        n_features = X.shape[1]
        
        # Initializing weights
        self.w_ = np.random.randn(n_features, 1)
        self.b_ = np.random.randn()
        self.errors_ = []
        
        # Fitting
        for _ in range(self.n_iter):
            error = 0
            #Update weights for each sample
            for xi, yi in zip(X, y):
                update = self.eta*(yi - self.predict(xi.reshape(1, n_features)))
                
                self.w_ += update*xi.reshape(n_features, 1)
                self.b_ += update
                
                error += int(update!=0.0)
            
            self.errors_.append(error)
        return self
    
    def predict(self, X):
        """
        Input:
            X: Array of size [n_samples, n_features]
                Input array vector
        Output:
            y_predict: Size [n_samples, 1]
                Predicted target values
        """
        Z = np.dot(X, self.w_) + self.b_ 
        y_predict = np.where(Z>=0.0, 1.0, -1.0)
        return y_predict
    
    
    
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