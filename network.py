import numpy as np
import random
from index import index

class Network():
    def __init__(self, neurons):
        self.neuron_layers = len(neurons)
        self.neurons = neurons
        self.biases = [np.random.uniform(low=0, high=1, size=(y, 1)) for y in neurons[1:]]
        self.weights = [np.random.uniform(low=0, high=1, size=(y,x)) for x, y in zip(neurons[:-1], neurons[1:])]

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def start_training(self, training_inputs, epochs, learning_rate, test_inputs=None):
        if test_inputs:
            test_inputs = list(test_inputs)
            n_test = len(test_inputs)

        length = len(training_inputs)
        for epoch in range(epochs):
            packed_inputs = [training_inputs[k:k+1] for k in range(length)]

            for single_input in packed_inputs:
                self.update_input(single_input, learning_rate)

            if test_inputs:
                print("Epoch", epoch, ":", self.identify_many(test_inputs))
            else:
                print("Epoch", epoch)

    def update_input(self, single_input, learning_rate):
        new_biase = [np.zeros(biase.shape) for biase in self.biases]
        new_weight = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in single_input:
            delta_new_biase, delta_new_weight = self.backprop(x, y)
            new_biase = [nb + dnb for nb, dnb in zip(new_biase, delta_new_biase)]
            new_weight = [nw + dnw for nw, dnw in zip(new_weight, delta_new_weight)]

        self.weights = [w-(learning_rate/len(single_input))*nw for w, nw in zip(self.weights, new_weight)]
        self.biases = [b-(learning_rate/len(single_input))*nb for b, nb in zip(self.biases, new_biase)]

    # x = entrada ; y = saida esperada
    def backprop(self,x,y):
        new_biase = [np.zeros(biase.shape) for biase in self.biases]
        new_weight = [np.zeros(weight.shape) for weight in self.weights]
        error_factor = 1
       
        activation = x
        activations = [x]
        zs = []

        # Here we are calculating feedforward function
        ##############################################################
        for biase, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + biase
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        ###############################################################
        
        #calculating and update error
        ############################################################### 
        # delta = fator erro
        delta = self.cost_derivative(activations[-1], y) * error_factor
        new_biase[-1] = delta
        new_weight[-1] = np.dot(delta, activations[-2].transpose())

        layer = 2
    
        z = zs[-layer]
        delta = np.dot(self.weights[-layer+1].transpose(), delta) * error_factor
        new_biase[-layer] = delta
        new_weight[-layer] = np.dot(delta, activations[-layer-1].transpose())
        #########################################################################

        return new_biase, new_weight

    def identify_many(self, training_inputs):
        return sum(int(self.identify(single_test)) for single_test in training_inputs)

    def identify(self, single_test, log=False):
        prediction = np.argmax(self.feedforward(single_test[0]))
        answer = np.argmax(single_test[1])
        if log:
            if prediction == answer:
                print("✅", index[answer], "→", index[prediction])
            else:
                print("❌", index[answer], "→", index[prediction])
        return prediction == answer

    # Retorna o vetor das derivadas parciais
    def cost_derivative(self, output_activations, y):
        return output_activations-y


