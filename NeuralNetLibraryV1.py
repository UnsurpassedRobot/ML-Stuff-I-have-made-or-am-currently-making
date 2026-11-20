import math

def sigmoid(z):
    y = 1 / (1 + math.exp(-z))
    return y

def ReLU(min, X):
    if min > X:
        y = 0
    else:
        y = X
    return y

class neuron:
    def __init__(self, learn_rate, layer_type, weights, activation_function):
        if layer_type == "input":
            self.w = [0] * weights
        elif layer_type == "hidden":
            self.b = 0
            self.w = [0] * weights
        elif layer_type == "output":
            self.b = 0
        self.type = layer_type
        self.l = learn_rate
        self.a = activation_function
    def forward(self, X):
        if self.type == "input":
            y = []
            for i in range(len(self.w)):
                y.append(X * self.w[i])
            return y
        elif self.type == "hidden":
            y = []
            z = sum(X)
            z += self.b
            if self.a == "sigmoid":
                z = sigmoid(z)
            elif self.a == "ReLU":
                z = ReLU(0, z)
            for i in range(len(self.w)):
                y.append(z * self.w[i])
            return y
        elif self.type == "output":
            y = sum(X)
            y += self.b
            if self.a == "sigmoid":
                y = sigmoid(y)
            elif self.a == "ReLU":
                y = ReLU(0, y)
            return y
    def update(self, wSlopes, bSlope):
        if self.type == "hidden" or self.type == "output":
            self.b -= self.l * bSlope
        if self.type == "input" or self.type == "hidden":
            for i in range(len(self.w)):
                self.w[i] -= self.l * wSlopes[i]