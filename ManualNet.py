from NeuralNetLibrary import neuron

epochs = 1000

hidden_layer_1 = []
hidden_layer_2 = []
output = neuron(0.001, 4, "sigmoid")

for i in range(4):
  hidden_layer_1.append(neuron(0.001, 2, "ReLU"))
  hidden_layer_2.append(neuron(0.001, 4, "ReLU"))

def predict(X):
  first = []
  for i in range(len(hidden_layer_1)):
    first.append(hidden_layer_1[i].forward(X))
  second = []
  for i in range(len(hidden_layer_2)):
    second.append(hidden_layer_2[i].forward(first))
  prediction = output.forward(second)
  return prediction

def train(X, y):
  for e in range(epochs):
    for i in range(len(X)):
      prediction = predict(X[i])
      error = y[i] - prediction