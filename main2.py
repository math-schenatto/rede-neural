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

    def identify(self,single_line):
        prediction = self.feedforward(single_line[0])
        answer = single_line[1]
        if np.array_equal(answer[1],prediction[1]):
            print("Acertou!")
        else:
            print("A resposta é:", answer, "A rede respondeu:", prediction)


    def cost_derivative(self, output_activations, y):
        # Retorna o vetor das derivadas parciais
        return output_activations-y


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

neuronios_entrada = 9
neuronios_intermediario = 6
neuronios_saida = 3

treinamento = [
    ('010111010','001'),
    ('000111000','010'),
    ('101010101','100')
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

    training_inputs.append((np.reshape(entrada_array, (neuronios_entrada, 1)), np.reshape(saida_array, (neuronios_saida, 1))))

# dtype=float32
# Imports
#import mnist_loader

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#training_data = list(training_data)
#import  IPython as ipy
#ipy.embed()

rede = Network([neuronios_entrada,neuronios_intermediario,neuronios_saida])
rede.SGD(training_inputs, 1000, mini_batch_size=20, eta=0.5, test_data=None)


teste = [
    ('010111010','001'),
    ('010111011','001'),
]
test_inputs = []
for linha in teste:
    entrada = []
    saida = []

    for e in linha[0]:
        entrada.append(e)
    for s in linha[1]:
        saida.append(s)

    entrada_array = np.array(entrada, dtype=np.float32)
    saida_array = np.array(saida, dtype=np.float32)

    test_inputs.append((np.reshape(entrada_array, (neuronios_entrada, 1)), np.reshape(saida_array, (neuronios_saida, 1))))

single_test = test_inputs[0]

#print(rede.num_layers, rede.biases, rede.weights)
