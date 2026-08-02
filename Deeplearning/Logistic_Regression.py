import math

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        n = len(X)

        for epoch in range(self.epochs):
            dw = 0
            db = 0
            loss = 0

            for i in range(n):
                z = self.w * X[i] + self.b
                y_pred = self.sigmoid(z)

                error = y_pred - y[i]

                dw += error * X[i]
                db += error

                # cross entropy loss
                loss += -(y[i]*math.log(y_pred + 1e-9) + (1-y[i])*math.log(1-y_pred + 1e-9))

            dw /= n
            db /= n
            loss /= n

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        return [self.sigmoid(self.w * x + self.b) for x in X]

    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]
    
X = [1, 2, 3, 4, 5]
y = [0, 0, 0, 1, 1]  # phân loại

model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X, y)

print("w =", model.w)
print("b =", model.b)

print("X test:", [1.5, 3.5, 5])
print("Prob:", model.predict_proba([1.5, 3.5, 5]))
print("Class:", model.predict([1.5, 3.5, 5]))