import math

class PCA:
    """
    Principal Component Analysis (PCA) – giảm chiều dữ liệu.
    Dùng power iteration + deflation để tìm các thành phần chính.
    """
    def __init__(self, n_components):
        self.n_components = n_components   # số chiều sau khi giảm
        self.components = None             # các vector riêng (thành phần chính)
        self.mean = None                   # trung bình mỗi chiều
        self.explained_variance = None

    def fit(self, X):
        """
        X: list of lists (mỗi hàng là một điểm dữ liệu, mỗi cột là một đặc trưng)
        """
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0

        # 1. Tính trung bình mỗi feature
        self.mean = [0.0] * n_features
        for i in range(n_features):
            col_sum = sum(row[i] for row in X)
            self.mean[i] = col_sum / n_samples

        # 2. Center dữ liệu
        X_centered = []
        for row in X:
            centered_row = [row[j] - self.mean[j] for j in range(n_features)]
            X_centered.append(centered_row)

        # 3. Tính ma trận hiệp phương sai (covariance matrix)
        # cov[i][j] = (1/(n-1)) * sum_{k} (X_centered[k][i] * X_centered[k][j])
        cov = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(i, n_features):
                s = 0.0
                for k in range(n_samples):
                    s += X_centered[k][i] * X_centered[k][j]
                cov[i][j] = cov[j][i] = s / (n_samples - 1)

        # 4. Tìm các vector riêng (eigenvectors) và trị riêng bằng power iteration
        self.components = []
        self.explained_variance = []
        # Ma trận hiệp phương sai copy để deflation
        A = [row[:] for row in cov]

        for _ in range(self.n_components):
            # Power iteration để tìm vector riêng chính (ứng với trị riêng lớn nhất)
            v = [random_uniform(-1, 1) for __ in range(n_features)]
            # Chuẩn hóa vector
            norm = math.sqrt(sum(xi*xi for xi in v))
            v = [xi / norm for xi in v]

            for _iter in range(100):   # số lần lặp tối đa
                # v_new = A * v
                v_new = [0.0] * n_features
                for i in range(n_features):
                    s = 0.0
                    for j in range(n_features):
                        s += A[i][j] * v[j]
                    v_new[i] = s
                # Tính trị riêng (Rayleigh quotient) trước khi chuẩn hóa
                eigenval = sum(v_new[i] * v[i] for i in range(n_features))
                # Chuẩn hóa v_new
                norm_new = math.sqrt(sum(xi*xi for xi in v_new))
                if norm_new < 1e-12:
                    break
                v = [xi / norm_new for xi in v_new]
                # Kiểm tra hội tụ (thay đổi rất nhỏ) – có thể bỏ qua để đơn giản

            # Lưu lại vector riêng (chính xác hơn 1 chút: tính v chính xác hơn)
            # Gán trị riêng
            self.components.append(v[:])
            self.explained_variance.append(eigenval)

            # Deflation: loại bỏ thành phần vừa tìm khỏi ma trận A
            # A = A - eigenval * (v * v^T)
            for i in range(n_features):
                for j in range(n_features):
                    A[i][j] -= eigenval * v[i] * v[j]

    def transform(self, X):
        """Chiếu dữ liệu X lên các thành phần chính"""
        if self.components is None:
            raise RuntimeError("Chưa fit PCA. Gọi fit trước.")
        # Center dữ liệu
        X_centered = []
        for row in X:
            centered = [row[j] - self.mean[j] for j in range(len(self.mean))]
            X_centered.append(centered)
        # Project lên các components
        result = []
        for row in X_centered:
            proj = []
            for comp in self.components:
                dot = sum(row[i] * comp[i] for i in range(len(row)))
                proj.append(dot)
            result.append(proj)
        return result

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ------------------- HÀM PHỤ TRỢ (vì không dùng random.uniform?) -------------------
def random_uniform(a, b):
    """Tạo số thực ngẫu nhiên trong [a,b] dùng linear congruential generator đơn giản"""
    # Dùng seed cố định để kết quả reproducible (có thể thay bằng random.random nếu muốn)
    # Ở đây tôi dùng seed tĩnh, bạn có thể bỏ qua, dùng random module được phép.
    import random
    return random.uniform(a, b)


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Dữ liệu 2D (dễ hình dung)
    X = [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ]

    print("Dữ liệu gốc (10 mẫu, 2 chiều):")
    for row in X:
        print([round(x, 2) for x in row])

    pca = PCA(n_components=1)   # Giảm xuống 1 chiều
    X_reduced = pca.fit_transform(X)

    print("\nSau khi giảm xuống 1 chiều (thành phần chính đầu tiên):")
    for val in X_reduced:
        print(round(val[0], 4))

    print("\nThành phần chính (vector riêng):", [round(x, 4) for x in pca.components[0]])
    print("Phương sai giải thích:", round(pca.explained_variance[0], 4))