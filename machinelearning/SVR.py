class SVR:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epsilon=0.5, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        n = len(X)

        for epoch in range(self.epochs):
            for i in range(n):
                x_i = X[i]
                y_i = y[i]

                y_pred = self.w * x_i + self.b
                error = y_i - y_pred

                if abs(error) <= self.epsilon:
                    # trong vùng epsilon → không phạt
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # ngoài vùng epsilon → bị phạt
                    sign = -1 if error > 0 else 1
                    dw = 2 * self.lambda_param * self.w + sign * x_i
                    db = sign

                self.w -= self.lr * dw
                self.b -= self.lr * db

            if epoch % 100 == 0:
                loss = sum(max(0, abs(y[i] - (self.w*X[i] + self.b)) - self.epsilon) for i in range(n)) / n
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return [self.w * x + self.b for x in X]

X = [1, 2, 3, 4, 5]
y = [1.2, 2.1, 2.9, 4.2, 5.1]  # gần y = x

model = SVR(learning_rate=0.01, epsilon=0.3, epochs=1000)
model.fit(X, y)

print("w =", model.w)
print("b =", model.b)

print("Predict:", model.predict([6, 7]))