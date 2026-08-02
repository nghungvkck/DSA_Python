class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # regularization
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        n = len(X)

        for epoch in range(self.epochs):
            for i in range(n):
                x_i = X[i]
                y_i = y[i]

                condition = y_i * (self.w * x_i + self.b) >= 1

                if condition:
                    # đúng → chỉ regularization
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # sai → bị phạt
                    dw = 2 * self.lambda_param * self.w - y_i * x_i
                    db = -y_i

                self.w -= self.lr * dw
                self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, w={self.w:.4f}, b={self.b:.4f}")

    def predict(self, X):
        result = []
        for x in X:
            val = self.w * x + self.b
            result.append(1 if val >= 0 else -1)
        return result

X = [1, 2, 3, 4, 5]
y = [-1, -1, -1, 1, 1]

model = SVM(learning_rate=0.01, epochs=1000)
model.fit(X, y)

print("w =", model.w)
print("b =", model.b)

print("Predict:", model.predict([1.5, 3.5, 5]))