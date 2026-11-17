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
  def __init__(self, leanr_rate, features, activations):
    self.bias = 0
    self.weights = [0] * features
    self.l = learn_rate
    self.a = activation
    if self.a == "ReLU":
      self.min = float(input("min value for ReLU"))
  
