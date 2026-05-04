import random

class Perceptron:
    """
    Perceptron cho phân loại nhị phân (0/1).
    Không dùng numpy, sklearn,...
    """
    def __init__(self, learning_rate=0.1, n_iter=100):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weights = None   # danh sách trọng số
        self.bias = None

    def _activation(self, z):
        """Hàm bước nhảy (step function)"""
        return 1 if z >= 0 else 0

    def predict_single(self, x):
        """Dự đoán một mẫu x (list)"""
        # Tính tổng trọng số: w0*x0 + w1*x1 + ... + bias
        z = self.bias
        for i in range(len(x)):
            z += self.weights[i] * x[i]
        return self._activation(z)

    def fit(self, X, y):
        """
        Huấn luyện Perceptron.
        X: list of lists (mỗi phần tử là một mẫu)
        y: list nhãn (0 hoặc 1)
        """
        n_features = len(X[0])
        # Khởi tạo trọng số ngẫu nhiên nhỏ và bias = 0
        random.seed(42)
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = 0.0

        for epoch in range(self.n_iter):
            errors = 0
            for i in range(len(X)):
                # Dự đoán
                pred = self.predict_single(X[i])
                # Tính lỗi
                error = y[i] - pred
                if error != 0:
                    errors += 1
                    # Cập nhật trọng số và bias
                    for j in range(n_features):
                        self.weights[j] += self.lr * error * X[i][j]
                    self.bias += self.lr * error
            # Nếu không có lỗi nào, dừng sớm
            if errors == 0:
                print(f"Hội tụ tại epoch {epoch+1}")
                break

    def predict(self, X):
        """Dự đoán cho nhiều mẫu"""
        return [self.predict_single(x) for x in X]


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Dữ liệu cổng AND
    X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_and = [0, 0, 0, 1]

    # Dữ liệu cổng OR
    X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_or = [0, 1, 1, 1]

    print("=== Cổng AND ===")
    p = Perceptron(learning_rate=0.1, n_iter=10)
    p.fit(X_and, y_and)
    print("Trọng số:", p.weights)
    print("Bias:", p.bias)
    for x in X_and:
        print(f"Input: {x} -> Dự đoán: {p.predict_single(x)}")

    print("\n=== Cổng OR ===")
    p2 = Perceptron(learning_rate=0.1, n_iter=10)
    p2.fit(X_or, y_or)
    print("Trọng số:", p2.weights)
    print("Bias:", p2.bias)
    for x in X_or:
        print(f"Input: {x} -> Dự đoán: {p2.predict_single(x)}")