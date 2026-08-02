class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        n = len(X)

        for epoch in range(self.epochs):
            dw = 0
            db = 0

            for i in range(n):
                y_pred = self.w * X[i] + self.b
                error = y_pred - y[i]

                dw += error * X[i]
                db += error

            dw = (2/n) * dw
            db = (2/n) * db

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                loss = sum((self.w * X[i] + self.b - y[i])**2 for i in range(n)) / n
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return [self.w * x + self.b for x in X]
    

X = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]  # y = 2x

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

print("w =", model.w)
print("b =", model.b)

print("Predict:", model.predict([6, 7]))