import math
import random

class LDA:
    """
    Linear Discriminant Analysis (LDA) – giảm chiều có giám sát.
    Tìm các vector (thành phần phân biệt) sao cho tối đa phương sai giữa các lớp / phương sai trong nội bộ lớp.
    """
    def __init__(self, n_components):
        self.n_components = n_components   # số chiều sau khi chiếu (<= số lớp - 1)
        self.scalings = None               # các vector chiếu (mỗi cột là một thành phần)
        self.means = []                    # vector trung bình của từng class
        self.class_labels = []             # danh sách các class label duy nhất
        self.global_mean = None

    def fit(self, X, y):
        """
        X: list of lists (mỗi hàng là một mẫu, mỗi cột là đặc trưng)
        y: list of labels (cùng độ dài X)
        """
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        # Tìm các class duy nhất
        labels = sorted(set(y))
        self.class_labels = labels
        n_classes = len(labels)

        # 1. Tính trung bình toàn cục (global mean)
        self.global_mean = [0.0] * n_features
        for row in X:
            for j in range(n_features):
                self.global_mean[j] += row[j]
        for j in range(n_features):
            self.global_mean[j] /= n_samples

        # 2. Tính trung bình từng lớp và within-class scatter matrix (Sw)
        Sw = [[0.0] * n_features for _ in range(n_features)]
        Sb = [[0.0] * n_features for _ in range(n_features)]
        class_means = []
        class_counts = []
        # Tính mean từng lớp
        for lbl in labels:
            class_samples = [X[i] for i in range(n_samples) if y[i] == lbl]
            cnt = len(class_samples)
            class_counts.append(cnt)
            mean_vec = [0.0] * n_features
            for sample in class_samples:
                for j in range(n_features):
                    mean_vec[j] += sample[j]
            for j in range(n_features):
                mean_vec[j] /= cnt
            class_means.append(mean_vec)

            # Within-class scatter: Sw += sum_{x in class} (x - mean_c)(x - mean_c)^T
            for sample in class_samples:
                diff = [sample[j] - mean_vec[j] for j in range(n_features)]
                for i in range(n_features):
                    for j in range(n_features):
                        Sw[i][j] += diff[i] * diff[j]

        # 3. Between-class scatter matrix (Sb)
        for idx, mean_c in enumerate(class_means):
            cnt = class_counts[idx]
            diff = [mean_c[j] - self.global_mean[j] for j in range(n_features)]
            for i in range(n_features):
                for j in range(n_features):
                    Sb[i][j] += cnt * diff[i] * diff[j]

        # 4. Giải bài toán eigenvalue tổng quát: Sb * v = lambda * Sw * v
        #    -> (inv(Sw) * Sb) * v = lambda * v
        #    Tính Sw_inv bằng phương pháp Gauss-Jordan
        try:
            Sw_inv = self._inverse_matrix(Sw)
        except ValueError:
            # Nếu Sw singular, dùng pseudo-inverse đơn giản (thêm regularization)
            print("Warning: Sw is singular, adding small regularization.")
            reg = 1e-6
            for i in range(n_features):
                Sw[i][i] += reg
            Sw_inv = self._inverse_matrix(Sw)

        # Ma trận A = Sw_inv * Sb
        A = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                s = 0.0
                for k in range(n_features):
                    s += Sw_inv[i][k] * Sb[k][j]
                A[i][j] = s

        # 5. Tìm các vector riêng (thành phần phân biệt) bằng power iteration + deflation
        #    Số chiều tối đa có thể lấy là min(n_features, n_classes-1)
        max_components = min(n_features, n_classes - 1)
        if self.n_components > max_components:
            print(f"n_components giảm từ {self.n_components} xuống {max_components} (do ràng buộc số lớp)")
            self.n_components = max_components

        self.scalings = []
        A_copy = [row[:] for row in A]   # copy để deflation
        for _ in range(self.n_components):
            # Power iteration
            v = [random.uniform(-1, 1) for _ in range(n_features)]
            # Chuẩn hóa
            norm = math.sqrt(sum(xi*xi for xi in v))
            v = [xi / norm for xi in v]
            for _ in range(200):   # số lần lặp
                # v_new = A * v
                v_new = [0.0] * n_features
                for i in range(n_features):
                    s = 0.0
                    for j in range(n_features):
                        s += A_copy[i][j] * v[j]
                    v_new[i] = s
                # Tìm trị riêng (ước lượng)
                eigenval = sum(v_new[i] * v[i] for i in range(n_features))
                # Chuẩn hóa v_new
                norm_new = math.sqrt(sum(xi*xi for xi in v_new))
                if norm_new < 1e-12:
                    break
                v = [xi / norm_new for xi in v_new]
                # Kiểm tra hội tụ (tùy chọn)
            self.scalings.append(v[:])
            # Deflation: A = A - lambda * v * v^T
            for i in range(n_features):
                for j in range(n_features):
                    A_copy[i][j] -= eigenval * v[i] * v[j]

    def transform(self, X):
        """Chiếu dữ liệu X lên các thành phần phân biệt"""
        if self.scalings is None:
            raise RuntimeError("Chưa fit LDA. Gọi fit trước.")
        result = []
        for row in X:
            proj = [sum(row[j] * self.scalings[c][j] for j in range(len(row))) for c in range(self.n_components)]
            result.append(proj)
        return result

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    # Hàm nghịch đảo ma trận (Gauss-Jordan) không dùng numpy
    def _inverse_matrix(self, mat):
        n = len(mat)
        # Tạo ma trận mở rộng [mat | I]
        aug = [mat[i][:] + [float(i == j) for j in range(n)] for i in range(n)]
        # Gauss-Jordan elimination
        for col in range(n):
            # Tìm pivot
            pivot = None
            for row in range(col, n):
                if abs(aug[row][col]) > 1e-12:
                    pivot = row
                    break
            if pivot is None:
                raise ValueError("Ma trận không khả nghịch")
            # Đổi dòng
            if pivot != col:
                aug[col], aug[pivot] = aug[pivot], aug[col]
            # Chuẩn hóa dòng pivot
            pivot_val = aug[col][col]
            for j in range(2*n):
                aug[col][j] /= pivot_val
            # Loại bỏ cột ở các dòng khác
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    if abs(factor) > 1e-12:
                        for j in range(2*n):
                            aug[row][j] -= factor * aug[col][j]
        # Trích phần nghịch đảo
        inv = [aug[i][n:] for i in range(n)]
        return inv


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Dữ liệu 2D, 3 lớp (phân biệt được)
    X = [
        [2, 3], [3, 4], [4, 5], [5, 6],   # class 0
        [8, 2], [9, 3], [10, 4], [11, 5], # class 1
        [5, 8], [6, 9], [7, 10], [8, 11]  # class 2
    ]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    print("Dữ liệu gốc (12 mẫu, 2 chiều):")
    for i, row in enumerate(X):
        print(f"  {row} -> class {y[i]}")

    lda = LDA(n_components=2)   # tối đa 2 chiều (số class-1 = 2)
    X_lda = lda.fit_transform(X, y)

    print("\nSau LDA (chiếu xuống 2 chiều):")
    for i, proj in enumerate(X_lda):
        print(f"  {[round(coord, 4) for coord in proj]} -> class {y[i]}")

    print("\nVector chiếu (các thành phần phân biệt):")
    for idx, comp in enumerate(lda.scalings):
        print(f"Component {idx+1}: {[round(x, 4) for x in comp]}")