import math

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        """
        max_depth: độ sâu tối đa (None = không giới hạn)
        min_samples_split: số mẫu tối thiểu để split một node
        min_impurity_decrease: độ giảm phương sai tối thiểu để chấp nhận split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def _variance(self, y):
        """Tính phương sai của mảng y (sử dụng công thức phương sai mẫu hoặc tổng thể)."""
        if len(y) == 0:
            return 0.0
        mean = sum(y) / len(y)
        var = sum((val - mean) ** 2 for val in y) / len(y)  # phương sai tổng thể
        return var

    def _mse_reduction(self, y, y_left, y_right):
        """Tính độ giảm MSE = var_parent - (n_left/n)*var_left - (n_right/n)*var_right."""
        var_parent = self._variance(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        if n_left == 0 or n_right == 0:
            return 0.0
        weighted_var_children = (n_left / n) * self._variance(y_left) + (n_right / n) * self._variance(y_right)
        return var_parent - weighted_var_children

    def _best_split(self, X, y):
        """Tìm split tốt nhất dựa trên giảm MSE."""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0

        for feat in range(n_features):
            # Lấy các giá trị feature khác nhau
            values = sorted(set(X[i][feat] for i in range(n_samples)))
            # Thử split tại trung điểm các giá trị liên tiếp
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i+1]) / 2.0
                left_idx = [idx for idx in range(n_samples) if X[idx][feat] <= threshold]
                right_idx = [idx for idx in range(n_samples) if X[idx][feat] > threshold]

                if len(left_idx) < self.min_samples_split or len(right_idx) < self.min_samples_split:
                    continue

                y_left = [y[idx] for idx in left_idx]
                y_right = [y[idx] for idx in right_idx]
                gain = self._mse_reduction(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = threshold
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        if best_gain < self.min_impurity_decrease:
            return None
        return best_feature, best_threshold, best_left_idx, best_right_idx

    def _build_tree(self, X, y, depth):
        """Xây dựng cây đệ quy."""
        n_samples = len(y)
        # Giá trị dự đoán tại node hiện tại (trung bình)
        leaf_value = sum(y) / n_samples if n_samples > 0 else 0.0

        # Điều kiện dừng
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return {'leaf': True, 'value': leaf_value, 'n_samples': n_samples}

        split = self._best_split(X, y)
        if split is None:
            return {'leaf': True, 'value': leaf_value, 'n_samples': n_samples}

        feature, threshold, left_idx, right_idx = split
        X_left = [X[i] for i in left_idx]
        y_left = [y[i] for i in left_idx]
        X_right = [X[i] for i in right_idx]
        y_right = [y[i] for i in right_idx]

        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree,
            'mse_gain': self._mse_reduction(y, y_left, y_right),
            'n_samples': n_samples,
            'value': leaf_value  # giá trị trung bình của node (có thể dùng để debug)
        }

    def fit(self, X, y):
        """Huấn luyện cây."""
        if len(X) != len(y):
            raise ValueError("Số lượng mẫu và nhãn không khớp.")
        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, node, x):
        """Dự đoán một mẫu."""
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        else:
            return self._predict_one(node['right'], x)

    def predict(self, X):
        """Dự đoán cho nhiều mẫu."""
        return [self._predict_one(self.tree, x) for x in X]

    def _print_tree(self, node, indent=""):
        """In cây ra màn hình."""
        if node['leaf']:
            print(f"{indent}-> Dự đoán = {node['value']:.4f} (số mẫu = {node['n_samples']})")
            return
        print(f"{indent}Nếu feature {node['feature']} <= {node['threshold']:.4f}: (gain = {node['mse_gain']:.4f})")
        self._print_tree(node['left'], indent + "  ")
        print(f"{indent}Ngược lại:")
        self._print_tree(node['right'], indent + "  ")

    def print_tree(self):
        """In cây ra màn hình."""
        if self.tree is None:
            print("Cây chưa được huấn luyện.")
        else:
            self._print_tree(self.tree)


# -------------------------------
# Ví dụ sử dụng
# -------------------------------
if __name__ == "__main__":
    # Dữ liệu mẫu: y = x1 + x2 + noise nhẹ
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
    y = [3, 5, 7, 9, 11, 13, 15, 17]  # x1 + x2
    # Thêm nhiễu nhẹ để có tình huống không hoàn hảo
    y = [val + 0.2 * (i % 2) for i, val in enumerate(y)]  # nhiễu nhỏ

    # Tạo cây hồi quy
    reg = DecisionTreeRegressor(max_depth=3, min_samples_split=2, min_impurity_decrease=0.0)
    reg.fit(X, y)

    print("Cây hồi quy vừa huấn luyện:")
    reg.print_tree()

    # Dự đoán trên tập huấn luyện
    preds = reg.predict(X)
    print("\nThực tế :", [round(v, 2) for v in y])
    print("Dự đoán:", [round(v, 2) for v in preds])

    # Dự đoán mẫu mới
    X_new = [[2, 2], [5, 5]]
    print("\nDự đoán mẫu mới:", [round(v, 2) for v in reg.predict(X_new)])