from ModelPrototypesV5 import LogisticRegression
import math

model = LogisticRegression(10000, 0.1)

X = [[0, 1], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1, 0]]

y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

model.train(X, y)

print(model.predict(X))
print(y)