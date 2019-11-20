import numpy as np
import random
from indexcompleto import index
from data import get_test_data, get_training_data
import time

class Network():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.uniform(low=0, high=1, size=(y, 1)) for y in sizes[1:]]
        self.weights = [np.random.uniform(low=0, high=1, size=(y,x)) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, tax_aprendizado, test_data=None):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            #random.shuffle(training_data)
            entradas = [training_data[k:k+1] for k in range(0,n, 1)]

            for entrada in entradas:
                self.update_entrada(entrada, tax_aprendizado)

            if test_data:
                pass
                #print("Epoch {} : {} /".format(j, self.evaluate(test_data), n_test))
            else:
                pass
                #print("Epoch {} finalizada".format(j))

    def update_entrada(self, entrada, tax_aprendizado):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in entrada:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(tax_aprendizado/len(entrada))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(tax_aprendizado/len(entrada))*nb for b, nb in zip(self.biases, nabla_b)]

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
            import IPython as ipy 
            ipy.embed()
            time.sleep(1)
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

        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def identify(self,single_line):
        prediction = np.argmax(self.feedforward(single_line[0]))
        answer = np.argmax(single_line[1])
        if prediction == answer:
            print("Acertou! ✅")
        else:
            print("Errou! ❌")
        print("A resposta é:", index[answer], "A rede respondeu:", index[prediction])
        # print(self.feedforward(single_line[0]))
        # print(answer)


    def cost_derivative(self, output_activations, y):
        # Retorna o vetor das derivadas parciais
        return output_activations-y


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

neuronios_entrada = 48
neuronios_intermediario = 20
neuronios_saida = 36

rede = Network([neuronios_entrada,neuronios_intermediario, neuronios_saida])

# np.save('weights3.npy', rede.weights)
#saved_weights = np.load("weights2.npy",mmap_mode=None, allow_pickle=True)
#rede.weights = saved_weights

training_inputs = get_training_data()
test_input      = get_test_data()
print(rede.weights[-1])
print('------------------------------------')
print(rede.biases[-1])
rede.SGD(training_inputs, 10000,  tax_aprendizado=0.9, test_data=test_input)

print(" \n TREINAMENTO:")
for single_test in training_inputs:
    rede.identify(single_test)

print("\n TESTE:")
for x, single_test in enumerate(test_input):
    if x == 0:
        print( "\nTESTE:" )
    if x == 34:
        print( "\nRUIDO MÍNIMO:" )
    if x == 54:
        print( "\nRUIDO MÉDIO:" )
    if x == 74:
        print( "\nRUIDO AVANÇADO:" )
    if x == 94:
        print( "\nNÃO FAZEM PARTE:" )
    rede.identify(single_test)

print(rede.weights[-1])
print('------------------------------------')
print(rede.biases[-1])