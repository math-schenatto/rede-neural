import numpy as np
import random
class Network():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {} : {} /".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} finalizada".format(j))

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self,x,y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x

        # Lista para armazenar todas as ativações, camada por camada
        activations = [x]

        # lista para armazenar todos os vetores z, camada por camada
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        # Retorna o número de entradas de teste para as quais a rede neural
        # produz o resultado correto. Note que a saída da rede neural
        # é considerada o índice de qualquer que seja
        # neurônio na camada final que tenha a maior ativação.

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        # Retorna o vetor das derivadas parciais
        return output_activations-y


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

treinamento = [
    ('011110110011100001100001111111100001100001100001', '000000000000000000000000010000000000'),
    ('111110100011100001100001111110100001100011111110', '000000000000000000000000100000000000'),
    ('111111100001100000100000100000100000100001111111', '000000000000000000000001000000000000'),
    ('111110100011100001100001100001100001100011111110', '000000000000000000000010000000000000'),
    ('111111100000100000100000111110100000100000111111', '000000000000000000000100000000000000'),
    ('111111100000100000100000111110100000100000100000', '000000000000000000001000000000000000'),
    ('111111100000100000100000100111100001100001111111', '000000000000000000010000000000000000'),
    ('100001100001100001100001111111100001100001100001', '000000000000000000100000000000000000'),
    ('111111001100001100001100001100001100001100111111', '000000000000000001000000000000000000'),
    ('000001000001000001000001000001100001100001111111', '000000000000000010000000000000000000'),
    ('100011100110101100111000111000101100100110100011', '000000000000000100000000000000000000'),
    ('100000100000100000100000100000100000100000111111', '000000000000001000000000000000000000'),
    ('110011111111101101101101100001100001100001100001', '000000000000010000000000000000000000'),
    ('110001111001101001101101100101100101100111100011', '000000000000100000000000000000000000'),
    ('111111100001100001100001100001100001100001111111', '000000000001000000000000000000000000'),
    ('111111100001100001100001111111100000100000100000', '000000000010000000000000000000000000'),
    ('011110110011100001101101100101110111011110000011', '000000000100000000000000000000000000'),
    ('111111100001100001111111101100100110100011100001', '000000001000000000000000000000000000'),
    ('111111100000100000100000111111000001000001111111', '000000010000000000000000000000000000'),
    ('111111001100001100001100001100001100001100001100', '000000100000000000000000000000000000'),
    ('100001100001100001100001100001100001100001111111', '000001000000000000000000000000000000'),
    ('100001100001110011110011010010011110001100001100', '000010000000000000000000000000000000'),
    ('100001100001100001100001101101101101101101110011', '000100000000000000000000000000000000'),
    ('100001110011011110001100001100011110110011100001', '001000000000000000000000000000000000'),
    ('100001100001110011011110001100001100001100001100', '010000000000000000000000000000000000'),
    ('111111000011000110001100001100011000110000111111', '100000000000000000000000000000000000')
]
training_inputs = []
for linha in treinamento:
    entrada = []
    saida = []

    for e in linha[0]:
        entrada.append(e)
    for s in linha[1]:
        saida.append(s)

    entrada_array = np.array(entrada, dtype=np.float32)
    saida_array = np.array(saida, dtype=np.float32)

    training_inputs.append((np.reshape(entrada_array, (48, 1)), np.reshape(saida_array, (36, 1))))


# entrada = 48
# intermediario = 42
# saida = 36
# dtype=float32
# Imports
#import mnist_loader

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#training_data = list(training_data)
#import  IPython as ipy
#ipy.embed()

rede = Network([48,42,36])
rede.SGD(training_inputs, 1000, mini_batch_size=20, eta=0.5, test_data=None)

#print(rede.num_layers, rede.biases, rede.weights)

