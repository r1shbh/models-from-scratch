import numpy as np


class LeNet(object):
    """
    Implementing CNN from scratch in numpy
    """

    def __init__(self, epochs=5, learning_rate=0.01, batch_size=32, optimizer="sgd", input_size=(32, 32, 3)):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.input_size = input_size

        self.hparams = {"input": {"n_H": self.input_size[0], "n_W": self.input_size[1], "n_C": self.input_size[2]},
                        "layer1": {"type": "conv", "filter": 5, "pad": 0, "stride": 1, "num_filters": 6,
                                   "activation": "relu"},
                        "layer2": {"type": "conv", "filter": 2, "pad": 0, "stride": 2, "num_filters": 6,
                                   "activation": "relu"},
                        "layer3": {"type": "conv", "filter": 5, "pad": 0, "stride": 1, "num_filters": 16,
                                   "activation": "relu"},
                        "layer4": {"type": "pool", "filter": 2, "pad": 0, "stride": 2, "activation": "relu"},
                        "layer5": {"type": "fc", "dense": 120, "activation": "relu"},
                        "layer6": {"type": "fc", "dense": 84, "activation": "relu"},
                        "layer7": {"type": "fc", "dense": 10, "activation": "softmax"}}

        self.params = {}

        ## Output dict
        self.outputs = {"output0": self.input_size}

        self.A = []

    def initialize_weights(self):
        """
        It creates a dict of output layers for the corresponding model
        :return:
        """

        layer = 1
        while "layer" + str(layer) in self.hparams.keys():

            self.A.append(0)
            if self.hparams["layer" + str(layer)]["type"] == "conv":
                # What is the input coming to this layer
                input = self.outputs["output" + str(layer - 1)]

                # Extracting hyperparameters
                f = self.hparams["layer" + str(layer)]["filter"]
                p = self.hparams["layer" + str(layer)]["pad"]
                s = self.hparams["layer" + str(layer)]["stride"]

                # Defining output
                nH_next = (input[0] - f + 2 * p) // s + 1
                nW_next = (input[1] - f + 2 * p) // s + 1
                nC_next = self.hparams["layer" + str(layer)]["num_filters"]

                # Writing output
                self.outputs["output" + str(layer)] = (nH_next, nW_next, nC_next)

                # Initialize weights
                self.params["layer" + str(layer) + "_conv"] = {"W": 0.01 * np.random.randn(f, f, input[2], nC_next),
                                                               "b": np.zeros((1, 1, 1, nC_next))}

            elif self.hparams["layer" + str(layer)]["type"] == "fc":
                # Input
                input = self.outputs["output" + str(layer - 1)]
                input_flat = np.prod(input)

                # Output
                output = self.hparams["layer" + str(layer)]["dense"]

                # Writing output
                self.outputs["output" + str(layer)] = output

                # Initialize weights
                self.params["layer" + str(layer) + "_fc"] = {"W": 0.01 * np.random.randn(input_flat, output),
                                                             "b": np.zeros((1, output))}

            elif self.hparams["layer" + str(layer)]["type"] == "pool":

                # Defining input (i.e ouput from previous layer)
                input = self.outputs["output" + str(layer - 1)]

                # Extracting hyperparameters
                f = self.hparams["layer" + str(layer)]["filter"]
                p = self.hparams["layer" + str(layer)]["pad"]
                s = self.hparams["layer" + str(layer)]["stride"]

                # Defining Output
                nH_next = (input[0] - f + 2 * p) // s + 1
                nW_next = (input[1] - f + 2 * p) // s + 1
                nC_next = input[2]

                # Writing output
                self.outputs["output" + str(layer)] = (nH_next, nW_next, nC_next)

            self.A.append(0)

            layer += 1

        return self

    def conv_slice(self, a_pad, col, row, filter, stride):

        start_col = stride * col
        end_col = start_col + filter

        start_row = stride * row
        end_row = start_row + filter

        a_slice = a_pad[start_row:end_row, start_col:end_col, :]

        return a_slice

    def conv2d(self, a_prev, layer):
        """
        To implement a single forward convolution step

        Input
        ---------
        a_prev: An array of shape (m, n_H, n_W, n_C_prev)

        params: A dict containing parameters of that layer
            params = {"W": ... , "b": ...}
            W: Array of shape (f, f, n_C_prev, n_C)
            b: Array of shape (1,1,1, n_C)

        hparams: A dict of hyperparameters for that layer
            filter size (f)
            strides (s)
            padding (p)
            no of filters (n_C)

        Output
        ----------
        a: Array of shape (m, n_H_next, n_W_next, n_C)
        """

        # Unfolding a_prev
        m, n_H, n_W, n_C_prev = a_prev.shape

        # Unfolding hyperparameters
        stride = self.hparams["layer" + str(layer)]["stride"]
        f = self.hparams["layer" + str(layer)]["filter"]
        pad = self.hparams["layer" + str(layer)]["pad"]

        # Unfolding parameters
        W = self.params["layer" + str(layer) + "_conv"]["W"]
        b = self.params["layer" + str(layer) + "_conv"]["b"]

        # Padding the a_prev (m, n_H, n_W, n_C_prev)
        a_pad = np.pad(a_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

        # Defining a_next
        nH_next, nW_next, nC_next = self.outputs["output" + str(layer)]

        a_next = np.zeros((m, nH_next, nW_next, nC_next))

        # Filling up a_next

        for sample in np.arange(m):
            for num_filter in np.arange(nC_next):
                for row in np.arange(nW_next):
                    for col in np.arange(nH_next):
                        a_slice_prev = self.conv_slice(a_pad[sample, :, :, :], col, row, f, stride)

                        a_next[sample, row, col, num_filter] = np.sum(W[:, :, :, num_filter] * a_slice_prev) + b[:, :,
                                                                                                               :,
                                                                                                               num_filter]

        return a_next

    def max_pool(self, a_prev, layer):

        # Unfolding a_prev
        m, n_H, n_W, n_C_prev = a_prev.shape

        # Unfolding hyperparameters
        stride = self.hparams["layer" + str(layer)]["stride"]
        f = self.hparams["layer" + str(layer)]["filter"]
        pad = self.hparams["layer" + str(layer)]["pad"]

        # Padding the a_prev (m, n_H, n_W, n_C_prev)
        a_pad = np.pad(a_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

        # Defining a_next
        nH_next, nW_next, nC_next = self.outputs["output" + str(layer)]

        a_next = np.zeros((m, nH_next, nW_next, nC_next))

        # Filling up a_next

        for sample in np.arange(m):
            for num_filter in np.arange(nC_next):
                for row in np.arange(nW_next):
                    for col in np.arange(nH_next):
                        a_slice_prev = self.conv_slice(a_pad[sample, :, :, :], col, row, f, stride)

                        a_next[sample, row, col, num_filter] = np.max(a_slice_prev[:, :, num_filter])

        return a_next

    def fully_connected(self, a_prev, layer):

        m = a_prev.shape[0]
        other_dims = np.prod(a_prev.shape[1:])
        a_flat = a_prev.reshape(m, other_dims)

        W = self.params["layer" + str(layer) + "_fc"]["W"]
        b = self.params["layer" + str(layer) + "_fc"]["b"]

        a_next = np.dot(a_flat, W) + b

        return a_next

    def activation(self, a_prev, activation="relu"):

        assert (activation == "relu" or activation == "softmax")

        if activation == "relu":
            a_prev[a_prev < 0] = 0

        elif activation == "softmax":
            a_prev = np.exp(a_prev) / np.sum(np.exp(a_prev))

        return a_prev

    def forward_propogate(self, X_batch):

        assert ((X_batch.shape[1], X_batch.shape[2], X_batch.shape[3] == (self.input_size)))

        self.A[0] = X_batch

        layer = 1

        while "layer" + str(layer) in self.hparams.keys():

            if self.hparams["layer" + str(layer)]["type"] == "conv":
                self.A[layer] = self.conv2d(self.A[layer - 1], layer)

            elif self.hparams["layer" + str(layer)]["type"] == "pool":
                self.A[layer] = self.max_pool(self.A[layer - 1], layer)

            elif self.hparams["layer" + str(layer)]["type"] == "fc":
                self.A[layer] = self.fully_connected(self.A[layer - 1], layer)

            self.A[layer] = self.activation(self.A[layer], self.hparams["layer" + str(layer)]["activation"])

            layer += 1

        output = self.A[layer - 1]

        return output
