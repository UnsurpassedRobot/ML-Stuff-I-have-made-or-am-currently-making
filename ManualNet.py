from NeuralNetLibraryV1 import neuron

def flip(X):
    flipped = []
    for length in range(len(X[0])):
        temp = []
        for height in range(len(X)):
            temp.append(X[height][length])
        flipped.append(temp)
    return flipped

input_layer = []
hidden_layer_1 = []
hidden_layer_2 = []
output_layer = neuron(0.001, "output", "null", "sigmoid")

for i in range(2):
    input_layer.append(neuron(0.001, "input", 4, "null"))

for i in range(4):
    hidden_layer_1.append(neuron(0.001, "hidden", 4, "ReLU"))
    hidden_layer_2.append(neuron(0.001, "hidden", 1, "ReLU"))

def predict(X):
    one = []
    for i in range(2):
        one.append(input_layer[i].forward(X[i]))
    one = flip(one)
    two = []
    for i in range(4):
        two.append(hidden_layer_1[i].forward(one[i]))
    two = flip(two)
    three = []
    for i in range(4):
        three.append(hidden_layer_2[i].forward(two[i]))
    three = flip(three)
    final = output_layer.forward(three)
    return one, two, three, final

def train(X, y):
    for e in range(1000):
        for i in range(len(X)):
            one, two, three, prediction = predict(X[i])
            bSlope = (2 * (prediction - y[i])) * (prediction * (1 - prediction))
            wSlopes = []
            for i in range(len(three)):
                wSlopes.append((2 * (prediction - y[i])) * (prediction * (1 - prediction)) * three[i])
            output_layer.update(wSlopes, bSlope)
