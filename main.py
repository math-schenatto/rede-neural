import numpy as np
from data import input_neurons, hidden_neurons, output_neurons
from data import get_test_data, get_training_data
from network2 import Network
import matplotlib.pyplot as plt


network = Network([input_neurons, hidden_neurons, output_neurons])
#saved_weights = np.load("weights.npy",mmap_mode=None, allow_pickle=True)
#network.weights = saved_weights

training_inputs = get_training_data()
test_inputs     = get_test_data()

network.SGD(training_inputs, 2000,  learning_rate=0.9, test_inputs=test_inputs)

print("\nRESULTADOS:")
for x, single_test in enumerate(test_inputs):
    if x == 0:
        print( "\nSEM RUIDOS:" )
    if x == 34:
        print( "\nRUIDO MÍNIMO:" )
    if x == 54:
        print( "\nRUIDO MÉDIO:" )
    if x == 74:
        print( "\nRUIDO AVANÇADO:" )
    if x == 94:
        print( "\nNÃO FAZEM PARTE:" )
    network.identify(single_test, log=True)

plt.ylabel('True positive rate')
plt.xlabel('False positive rate')

for character in network.metrics:
    # network.metrics[character]['fpr'].sort()
    # network.metrics[character]['tpr'].sort()
    plt.plot(network.metrics[character]['fpr'], network.metrics[character]['tpr'], label=character, marker='.')

# plt.plot(network.metrics['Z']['fpr'], network.metrics['Z']['tpr'], label='Z')
# plt.plot(network.metrics['J']['fpr'], network.metrics['J']['tpr'], label='J')
# plt.plot(network.metrics['U']['fpr'], network.metrics['U']['tpr'], label='U')
# plt.plot(network.metrics['Y']['fpr'], network.metrics['Y']['tpr'], label='Y')


plt.legend(ncol=3,loc='lower right')
plt.show()
