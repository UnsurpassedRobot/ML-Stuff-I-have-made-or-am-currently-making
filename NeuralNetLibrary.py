import math

def sigmoid(z):
  X = 1 / (1 + math.exp(-z))
  return X

def ReLU(min, X):
  if min > X:
    X = 0
  else:
    X = X
  return X

class neuron:
  def __init__(self, learn_rate, features, activations):
    self.b = 0
    self.w = [0] * features
    self.l = learn_rate
    self.a = activation
    if self.a == "ReLU":
      self.min = float(input("min value for ReLU"))
  def forward(self, X):
    prediction = self.b
    for i in range(len(self.w)):
      prediction += X[i] * self.w[i]
    if self.a == "sigmoid":
      prediction = sigmoid(prediction)
    elif self.a == "ReLU":
      prediction = ReLU(self.min, prediction)
    return prediction
  def update(self, bSlope, wSlopes):
    self.b -= bSlope * self.l
    for i in range(len(self.w)):
      self.w[i] -= wSlopes[i] * self.l 
