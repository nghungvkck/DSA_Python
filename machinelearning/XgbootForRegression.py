import math
import random

# -------------------------------
# Cây hồi quy hỗ trợ gradient & hessian (dùng cho regression)
# -------------------------------
class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=5, lambda_reg=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.root = None

    class Node:
        def __init__(self, leaf_value=None, feature_idx=None, threshold=None, left=None, right=None):
            self.leaf_value = leaf_value
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right

    def _compute_gain(self, G, H):
        """Tính gain = 0.5 * G^2/(H+lambda)"""
        return (G * G) / (H + self.lambda_reg)

    def _best_split(self, X, G, H):
        n_samples = len(X)
        G_total = sum(G)
        H_total = sum(H)
        gain_parent = self._compute_gain(G_total, H_total)

        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = []
        best_right_idx = []

        for feat in range(len(X[0])):
            # Sắp xếp theo giá trị feature
            data = sorted([(X[i][feat], G[i], H[i]) for i in range(n_samples)], key=lambda x: x[0])
            G_left = 0.0
            H_left = 0.0
            for i in range(n_samples - 1):
                G_left += data[i][1]
                H_left += data[i][2]
                G_right = G_total - G_left
                H_right = H_total - H_left

                if i+1 < self.min_samples_split or n_samples - (i+1) < self.min_samples_split:
                    continue
                if data[i][0] == data[i+1][0]:
                    continue

                gain_left = self._compute_gain(G_left, H_left)
                gain_right = self._compute_gain(G_right, H_right)
                gain = 0.5 * (gain_left + gain_right - gain_parent) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = (data[i][0] + data[i+1][0]) / 2.0
                    # Lưu chỉ số mẫu cho left/right (dựa trên thứ tự sau sắp xếp)
                    # Cách an toàn: dùng lại toàn bộ mẫu ban đầu để chia
        if best_feature is None:
            return None
        left_idx = [i for i in range(n_samples) if X[i][best_feature] <= best_threshold]
        right_idx = [i for i in range(n_samples) if X[i][best_feature] > best_threshold]
        return best_feature, best_threshold, left_idx, right_idx

    def _build_tree(self, X, G, H, depth):
        n_samples = len(X)
        G_total = sum(G)
        H_total = sum(H)
        leaf_value = -G_total / (H_total + self.lambda_reg)

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return self.Node(leaf_value=leaf_value)

        split = self._best_split(X, G, H)
        if split is None:
            return self.Node(leaf_value=leaf_value)

        feat, thresh, left_idx, right_idx = split
        if len(left_idx) == 0 or len(right_idx) == 0:
            return self.Node(leaf_value=leaf_value)

        X_left = [X[i] for i in left_idx]
        G_left = [G[i] for i in left_idx]
        H_left = [H[i] for i in left_idx]
        X_right = [X[i] for i in right_idx]
        G_right = [G[i] for i in right_idx]
        H_right = [H[i] for i in right_idx]

        left_node = self._build_tree(X_left, G_left, H_left, depth+1)
        right_node = self._build_tree(X_right, G_right, H_right, depth+1)
        return self.Node(feature_idx=feat, threshold=thresh, left=left_node, right=right_node)

    def fit(self, X, G, H):
        self.root = self._build_tree(X, G, H, 0)

    def _predict_one(self, node, x):
        if node.leaf_value is not None:
            return node.leaf_value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def predict(self, X):
        return [self._predict_one(self.root, x) for x in X]


# -------------------------------
# XGBoost Regressor (squared error)
# -------------------------------
class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.3, max_depth=3,
                 min_samples_split=5, lambda_reg=1.0, gamma=0.0, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.subsample = subsample
        self.trees = []
        self.initial_pred = None

    def _squared_loss_gradient_hessian(self, y_true, y_pred):
        """Gradient = pred - y_true, Hessian = 1 (cho squared loss 0.5*(y-p)^2)"""
        grad = [p - y for p, y in zip(y_pred, y_true)]
        hess = [1.0] * len(y_true)
        return grad, hess

    def fit(self, X, y):
        n_samples = len(X)
        # Khởi tạo dự đoán bằng trung bình y
        self.initial_pred = sum(y) / n_samples
        pred = [self.initial_pred] * n_samples

        for _ in range(self.n_estimators):
            # Tính gradient, hessian
            grad, hess = self._squared_loss_gradient_hessian(y, pred)

            # Subsample (nếu cần)
            if self.subsample < 1.0:
                idx = random.sample(range(n_samples), int(self.subsample * n_samples))
                X_sub = [X[i] for i in idx]
                grad_sub = [grad[i] for i in idx]
                hess_sub = [hess[i] for i in idx]
            else:
                X_sub, grad_sub, hess_sub = X, grad, hess

            # Xây dựng cây
            tree = RegressionTree(max_depth=self.max_depth,
                                  min_samples_split=self.min_samples_split,
                                  lambda_reg=self.lambda_reg,
                                  gamma=self.gamma)
            tree.fit(X_sub, grad_sub, hess_sub)

            # Dự đoán đầu ra của cây trên toàn bộ dữ liệu
            tree_pred = tree.predict(X)

            # Cập nhật dự đoán
            for i in range(n_samples):
                pred[i] += self.learning_rate * tree_pred[i]

            self.trees.append(tree)

    def predict(self, X):
        """Dự đoán giá trị hồi quy."""
        pred = [self.initial_pred] * len(X)
        for tree in self.trees:
            tree_pred = tree.predict(X)
            for i in range(len(X)):
                pred[i] += self.learning_rate * tree_pred[i]
        return pred


# -------------------------------
# Ví dụ sử dụng (hồi quy)
# -------------------------------
if __name__ == "__main__":
    # Dữ liệu mẫu: y = x1 + x2 + noise
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
    y = [3, 5, 7, 9, 11, 13, 15, 17]  # tương đương x1 + x2

    # Thêm nhiễu nhẹ
    y = [val + random.uniform(-0.5, 0.5) for val in y]

    # Huấn luyện XGBoost Regression
    model = XGBoostRegressor(n_estimators=10, learning_rate=0.2, max_depth=3,
                             lambda_reg=0.1, gamma=0.0, subsample=1.0)
    model.fit(X, y)

    # Dự đoán trên tập huấn luyện
    preds = model.predict(X)
    print("Thực tế:", [round(v, 2) for v in y])
    print("Dự đoán:", [round(v, 2) for v in preds])

    # Dự đoán mẫu mới
    X_new = [[2, 2], [5, 5]]
    print("Dự đoán mới:", [round(v, 2) for v in model.predict(X_new)])