from NeuralNetLbrary import neuron

hidden_layer_1 = []
hidden_layer_2 = []
output = neuron(0.001, 4, "sigmoid")

for i in range(4):
  hidden_layer_1.append(neuron(0.001, 57, "ReLU"))
  hidden_layer_2.append(neuron(0.001, 4, "ReLU"))

