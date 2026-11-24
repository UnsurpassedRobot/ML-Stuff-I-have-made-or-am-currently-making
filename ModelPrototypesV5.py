import math

class LinearRegression:
    def __init__(self, epochs, learn_rate, error):
        while epochs <= 0:
            epochs = int(input("invalid epochs, enter new value: "))
        while error != "MSE" and error != "mse" and error != "MAE" and error != "mae":
            error = input("invalid error metric, enter new error metric: ")
        self.e = epochs
        self.l = learn_rate
        self.error = error
        self.bias = 0
        self.weights = []
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            predict = self.bias
            for w in range(len(X[0])):
                predict += X[i][w] * self.weights[w]
            predictions.append(predict)
        return predictions
    def train(self, X, y):
        if len(X) == len(y):
            self.bias = 0
            self.weights = [0] * len(X[0])
            epochs = 0
            while self.e > epochs:
                predictions = self.predict(X)
                errors = []
                if self.error == "MSE" or self.error == "mse":
                    for i in range(len(X)):
                        errors.append((predictions[i] - y[i]) * 2)
                elif self.error == "MAE" or self.error == "mae":
                    for i in range(len(X)):
                        errors.append(predictions[i] - y[i]) * -1
                bSlope = sum(errors) / len(X)
                wSlopes = []
                for w in range(len(X[0])):
                    wErrors = []
                    for i in range(len(X)):
                        wErrors.append(X[i][w] * errors[i])
                    wSlopes.append(sum(wErrors) / len(X))
                self.bias -= bSlope * self.l
                for i in range(len(X[0])):
                    self.weights[i] -= wSlopes[i] * self.l
                epochs += 1
            print("error metric:",self.error)
            print("weight:", self.weights)
            print("bias:", self.bias)
        else:
            print("[INVALID NUMBER OF DATA VALUES]")

class LogisticRegression:
    def __init__(self, epochs, learn_rate):
        self.e = epochs
        self.l = learn_rate
        self.bias = 0
        self.weights = []
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            z = self.bias
            for w in range(len(X[0])):
                z += X[i][w] * self.weights[w]
            predict = 1/(1 + math.exp(-z))
            predictions.append(predict)
        return predictions
    def train(self, X, y):
        self.bias = 0
        self.weights = [0] * len(X[0])
        epochs = 0
        while epochs < self.e:
            predictions = self.predict(X)
            errors = []
            for i in range(len(X)):
                errors.append(predictions[i] - y[i])
            bSlope = sum(errors) / len(X)
            wSlopes = []
            for w in range(len(X[0])):
                wErrors = []
                for i in range(len(X)):
                    wErrors.append(X[i][w] * errors[i])
                wSlopes.append(sum(wErrors) / len(X))
            self.bias -= bSlope * self.l
            for i in range(len(self.weights)):
                self.weights[i] -= wSlopes[i] * self.l
            epochs += 1
