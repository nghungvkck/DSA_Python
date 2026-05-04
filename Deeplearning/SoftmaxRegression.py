import math
import random

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size  # None = full batch
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.n_features = None

    def _softmax(self, z):
        """z: ma trận (n_samples, n_classes)"""
        # Trừ max mỗi dòng để ổn định số
        max_vals = [max(row) for row in z]
        exp_z = [[math.exp(z[i][j] - max_vals[i]) for j in range(self.n_classes)] for i in range(len(z))]
        sum_exp = [sum(row) for row in exp_z]
        return [[exp_z[i][j] / sum_exp[i] for j in range(self.n_classes)] for i in range(len(z))]

    def _cross_entropy(self, y_true, y_pred):
        """y_true: one‑hot encoding (list of lists); y_pred: xác suất dự đoán"""
        n_samples = len(y_true)
        loss = 0.0
        for i in range(n_samples):
            for j in range(self.n_classes):
                if y_true[i][j] == 1:
                    loss += -math.log(y_pred[i][j] + 1e-15)
        return loss / n_samples

    def _to_one_hot(self, y):
        """Chuyển nhãn (0, 1, ..., n_classes-1) thành one‑hot encoding."""
        one_hot = [[0] * self.n_classes for _ in range(len(y))]
        for i, label in enumerate(y):
            one_hot[i][label] = 1
        return one_hot

    def fit(self, X, y):
        """
        X: list of lists, shape (n_samples, n_features)
        y: list of int, nhãn từ 0 đến n_classes-1
        """
        self.n_samples = len(X)
        self.n_features = len(X[0])
        self.n_classes = len(set(y))
        # Khởi tạo weights và bias
        self.weights = [[random.uniform(-0.01, 0.01) for _ in range(self.n_classes)] for _ in range(self.n_features)]
        self.bias = [0.0] * self.n_classes

        # Chuyển y sang one‑hot
        y_one_hot = self._to_one_hot(y)

        # Gradient descent
        for epoch in range(self.n_iterations):
            # Forward: tính điểm số (logits)
            logits = [[sum(X[i][f] * self.weights[f][c] for f in range(self.n_features)) + self.bias[c] 
                       for c in range(self.n_classes)] for i in range(self.n_samples)]
            probs = self._softmax(logits)

            # Tính loss (in để theo dõi)
            if epoch % 100 == 0:
                loss = self._cross_entropy(y_one_hot, probs)
                print(f"Epoch {epoch}, loss = {loss:.6f}")

            # Tính gradient
            grad_w = [[0.0] * self.n_classes for _ in range(self.n_features)]
            grad_b = [0.0] * self.n_classes

            for i in range(self.n_samples):
                for c in range(self.n_classes):
                    diff = probs[i][c] - y_one_hot[i][c]  # p - y
                    for f in range(self.n_features):
                        grad_w[f][c] += diff * X[i][f]
                    grad_b[c] += diff

            # Cập nhật trọng số
            for f in range(self.n_features):
                for c in range(self.n_classes):
                    self.weights[f][c] -= self.learning_rate * grad_w[f][c] / self.n_samples
            for c in range(self.n_classes):
                self.bias[c] -= self.learning_rate * grad_b[c] / self.n_samples

    def predict_proba(self, X):
        """Trả về xác suất cho từng lớp."""
        logits = [[sum(X[i][f] * self.weights[f][c] for f in range(self.n_features)) + self.bias[c] 
                   for c in range(self.n_classes)] for i in range(len(X))]
        return self._softmax(logits)

    def predict(self, X):
        """Trả về lớp có xác suất cao nhất."""
        probs = self.predict_proba(X)
        return [max(range(self.n_classes), key=lambda c: probs[i][c]) for i in range(len(X))]