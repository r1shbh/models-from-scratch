import numpy as np


class NeuralNetwork(object):
    
    def __init__(self, learning_rate=0.01, hidden_units=[2, 3], activation_functions = ['sigmoid', 'relu', 'sigmoid'],
                 epochs=200, random_state=2):
        """
        Initializing the model with the following hyperparameters
        -------------------------------
        
        Input
        -------------------------
        learning_rate: float
            Between 0.0 and 1.0
            
        hidden_layers: int
            Number of hidden layers (excluding input and output layers)
        
        hidden_units: 1d array
            Number of hidden units in each hidden layer. The output layers takes the shape of desired output (so not required)
        
        activation_functions: 1d array
            Activation function for each of the hidden layers AND Final Layer. Can be 'sigmoid', 'tanh' and 'relu' 
            If performing regression, 'relu' preferred for final layer.
        
        random_state: int
            Random seed for the random initializations
        """
        
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.hidden_layers = len(hidden_units)
        self.activation_functions  = activation_functions
        self.epochs = epochs
        self.random_state = random_state
        
        self.costs_ = []
    
    def initialize_weights(self, input_shape, output_shape):
        """
        Initializes the weights and biases based on recieved input and output shape.
        ----------------------------
        
        Input
        ------------------
        input_shape: 2-tuple
            tuple of shape (num_samples, num_features)
        output_shape: 2-tuple
            tuple of shape (num_samples, num_classes)
        """
        #Taking shapes of input and output
        self.num_samples, self.num_features = input_shape
        assert(self.num_samples == output_shape[0])
        
        if output_shape[1]:
            self.num_classes = output_shape[1]
        else:
            self.num_classes = 1
        
        #Setting the seed
        np.random.seed(self.random_state)
        
        #Now we can completely define output layer, i.e the last layer
        #Total number of layers will be hidden_layers + 1
        #It means, hidden_layers+1 number of weights and biases vector (incl. ouput 
        
        #Updating hidden_layers count
        self.num_layers = self.hidden_layers+1
        
        #Updating n_[l]
        self.n_ = [self.num_features] #n[0] = num_features
        self.n_.extend(self.hidden_units) #hidden layers
        self.n_.append(self.num_classes) #n[L] = num_classes
        
        #Defining weights and biases vectors
        self.weights = [np.array([0])]
        self.biases = [np.array([0])]
        
        #For any given layer, w.shape == n[l-1], n[l]
        for layer in range(1, self.num_layers+1):
            self.weights.append(np.random.randn(self.n_[layer-1], self.n_[layer])*0.001)
            self.biases.append(np.zeros((1, self.n_[layer])))
        
        return self
    
    
    def activation(self, Z, method='sigmoid'):
        """
        """
        assert(method in ['sigmoid', 'relu', 'tanh'])
        
        if method=='sigmoid':
            A = 1.0/(1 + np.exp(-Z))
        elif method=='relu':
            A = np.maximum(0, Z)
        elif method=='tanh':
            A = np.tanh(Z)
            
        return A
               
    
    def forward_propagation(self, X):
        """
        """
        #Initializing A and Z for each layer
        self.A = [np.array([0]) for _ in range(self.num_layers+1)]
        self.Z = [np.array([0]) for _ in range(self.num_layers+1)]
        
        self.A[0] = X
        
        for l in range(1, self.num_layers+1):
            self.Z[l] = np.dot(self.A[l-1], self.weights[l]) + self.biases[l]   #Z[l] of shape (m, n[l])
            self.A[l] = self.activation(self.Z[l], self.activation_functions[l-1])
            
        return self
    
    def compute_cost(self,  y):
        """
        """
        y_pred = self.A[-1]
        cost = (-1/self.num_samples)*np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
        
        self.costs_.append(cost)
        
        return self
    
    
    def activation_differentiation(self, l):
        """
        """
        method = self.activation_functions[l-1]
        
        assert(method in ['sigmoid', 'relu', 'tanh'])
        
        if method=='sigmoid':
            diff = self.A[l]*(1 - self.A[l])
        elif method=='relu':
            diff = np.where(self.Z[l] >=0, 1, 0)
        elif method=='tanh':
            diff = 1 - self.A[l]**2        
        
        return diff
        
    
    def backward_propagation(self, y):
        """
        """
        #Initializing dA, dZ, dW and dB
        self.dA = [np.zeros((self.A[l].shape)) for l in range(self.num_layers+1)]
        self.dZ = [np.zeros((self.Z[l].shape)) for l in range(self.num_layers+1)]
        self.dW = [np.zeros((self.weights[l].shape)) for l in range(self.num_layers+1)]
        self.dB = [np.zeros((self.biases[l].shape)) for l in range(self.num_layers+1)]
        
        self.dA[self.num_layers] = -(y/self.A[self.num_layers]) + ((1-y)/(1-self.A[self.num_layers]))
        
        for l in range(self.num_layers, 0 , -1):
            self.dZ[l] = self.dA[l]*self.activation_differentiation(l)  #dZ and dA of shape (m, n[l])
            self.dW[l] = np.dot(self.A[l-1].T, self.dZ[l])  #dW of shape (n[l-1], n[l])
            self.dB[l] = np.sum(self.dZ[l], axis = 0, keepdims= True)
            self.dA[l-1] = np.dot(self.dZ[l], self.weights[l].T)
            
        return
    
    def fit(self, X, y):
        """
        """
        
        #Initialize the parameters to get weights and biases
        self.initialize_weights(X.shape, y.shape)
        
        #Fitting
        for epoch in range(self.epochs):
            #Forward propogation. Fills A[l] and Z[l]
            self.forward_propagation(X)
            
            #Compute cost and append to costs_
            self.compute_cost(y)
            
            #Backward propagation. Fills dA[l], dZ[l], dW[l] and dB[l]
            self.backward_propagation(y)
            
            #Update parameters
            for l in range(1, self.num_layers+1):
                self.weights[l] -= self.learning_rate*self.dW[l]
                self.biases[l] -= self.learning_rate*self.dB[l]
                
        return self
    
    
    def predict(self, X):
        """
        """
        
        self.forward_propagation(X)
        y_pred = np.where(self.A[-1]>=0.5, 1, 0)
        
        return y_pred
        
    
            
        
    
    
    