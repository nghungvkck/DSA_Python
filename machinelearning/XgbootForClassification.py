import math
import random
from collections import Counter

# -------------------------------
# Cây hồi quy hỗ trợ gradient & hessian
# -------------------------------
class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=5, lambda_reg=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg  # L2 regularization
        self.gamma = gamma            # min loss reduction for split
        self.root = None

    class Node:
        def __init__(self, leaf_value=None, feature_idx=None, threshold=None, left=None, right=None):
            self.leaf_value = leaf_value   # giá trị tại lá (leaf weight)
            self.feature_idx = feature_idx # chỉ số feature dùng để split
            self.threshold = threshold     # ngưỡng split
            self.left = left
            self.right = right

    def _compute_gain(self, G, H):
        """Tính gain = 0.5 * (G^2/(H+lambda)) cho một node."""
        return (G * G) / (H + self.lambda_reg)

    def _best_split(self, X, G, H):
        """Tìm split tốt nhất dựa trên gain."""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        n_samples = len(X)
        # Tổng gradient và hessian toàn bộ node hiện tại
        G_total = sum(G)
        H_total = sum(H)
        gain_parent = self._compute_gain(G_total, H_total)

        for feat in range(len(X[0])):  # duyệt từng feature
            # Lấy giá trị feature và sắp xếp cùng gradient, hessian
            data = sorted([(X[i][feat], G[i], H[i]) for i in range(n_samples)], key=lambda x: x[0])
            G_left = 0.0
            H_left = 0.0
            for i in range(n_samples - 1):
                G_left += data[i][1]
                H_left += data[i][2]
                G_right = G_total - G_left
                H_right = H_total - H_left

                # Bỏ qua nếu một bên quá ít mẫu
                if i+1 < self.min_samples_split or n_samples - (i+1) < self.min_samples_split:
                    continue

                # Tránh split trùng giá trị feature
                if data[i][0] == data[i+1][0]:
                    continue

                gain_left = self._compute_gain(G_left, H_left)
                gain_right = self._compute_gain(G_right, H_right)
                gain = 0.5 * (gain_left + gain_right - gain_parent) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = (data[i][0] + data[i+1][0]) / 2.0
                    best_left_idx = [idx for idx in range(i+1)]
                    best_right_idx = [idx for idx in range(i+1, n_samples)]
                    # Lưu lại chỉ số mẫu để split sau (cần mapping lại thứ tự ban đầu)
                    # Tuy nhiên vì sắp xếp theo feature nên cần lưu chỉ số gốc.
                    # Cách đơn giản: trả về split trực tiếp chỉ dựa trên threshold.
        if best_feature is None:
            return None
        # Chia tập dữ liệu dựa trên threshold đã chọn
        left_idx = [i for i in range(n_samples) if X[i][best_feature] <= best_threshold]
        right_idx = [i for i in range(n_samples) if X[i][best_feature] > best_threshold]
        return best_feature, best_threshold, left_idx, right_idx

    def _build_tree(self, X, G, H, depth):
        """Xây dựng cây đệ quy."""
        n_samples = len(X)
        G_total = sum(G)
        H_total = sum(H)
        # Nếu đạt điều kiện dừng -> tạo lá
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = -G_total / (H_total + self.lambda_reg)
            return self.Node(leaf_value=leaf_value)

        split = self._best_split(X, G, H)
        if split is None:
            leaf_value = -G_total / (H_total + self.lambda_reg)
            return self.Node(leaf_value=leaf_value)

        feat, thresh, left_idx, right_idx = split
        if len(left_idx) == 0 or len(right_idx) == 0:
            leaf_value = -G_total / (H_total + self.lambda_reg)
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
        """Huấn luyện cây với gradient và hessian cho sẵn."""
        self.root = self._build_tree(X, G, H, 0)

    def _predict_one(self, node, x):
        """Dự đoán một mẫu."""
        if node.leaf_value is not None:
            return node.leaf_value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def predict(self, X):
        """Dự đoán output cho tập X."""
        return [self._predict_one(self.root, x) for x in X]


# -------------------------------
# XGBoost Classifier (binary classification)
# -------------------------------
class XGBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.3, max_depth=3,
                 min_samples_split=5, lambda_reg=1.0, gamma=0.0, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.subsample = subsample   # tỷ lệ lấy mẫu ngẫu nhiên (bagging)
        self.trees = []              # lưu các cây đã huấn luyện
        self.initial_pred = None     # giá trị khởi tạo (log-odds)

    def _sigmoid(self, x):
        """Hàm sigmoid."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _logistic_loss_gradient_hessian(self, y_true, y_pred_prob):
        """Tính gradient và hessian cho binary logistic loss."""
        # gradient = p - y
        # hessian = p * (1-p)
        grad = [p - y for p, y in zip(y_pred_prob, y_true)]
        hess = [p * (1.0 - p) for p in y_pred_prob]
        return grad, hess

    def fit(self, X, y):
        """
        X: list of list (features)
        y: list of int (0 hoặc 1)
        """
        n_samples = len(X)
        # Khởi tạo dự đoán: log-odds = log(mean(y)/(1-mean(y)))
        mean_y = sum(y) / n_samples
        self.initial_pred = math.log(mean_y / (1.0 - mean_y))
        # Dự đoán ban đầu (dạng margin)
        pred_margin = [self.initial_pred] * n_samples

        for _ in range(self.n_estimators):
            # Chuyển margin -> xác suất
            pred_prob = [self._sigmoid(p) for p in pred_margin]
            # Tính gradient và hessian
            grad, hess = self._logistic_loss_gradient_hessian(y, pred_prob)

            # Subsample (nếu cần)
            if self.subsample < 1.0:
                idx = random.sample(range(n_samples), int(self.subsample * n_samples))
                X_sub = [X[i] for i in idx]
                grad_sub = [grad[i] for i in idx]
                hess_sub = [hess[i] for i in idx]
            else:
                X_sub, grad_sub, hess_sub = X, grad, hess

            # Xây dựng cây hồi quy để dự đoán gradient
            tree = RegressionTree(max_depth=self.max_depth,
                                  min_samples_split=self.min_samples_split,
                                  lambda_reg=self.lambda_reg,
                                  gamma=self.gamma)
            tree.fit(X_sub, grad_sub, hess_sub)

            # Dự đoán giá trị của cây trên toàn bộ dữ liệu
            tree_pred = tree.predict(X)

            # Cập nhật margin: pred_margin += learning_rate * tree_pred
            for i in range(n_samples):
                pred_margin[i] += self.learning_rate * tree_pred[i]

            # Lưu cây lại
            self.trees.append(tree)

    def predict_proba(self, X):
        """Dự đoán xác suất lớp 1."""
        pred_margin = [self.initial_pred] * len(X)
        for tree in self.trees:
            tree_pred = tree.predict(X)
            for i in range(len(X)):
                pred_margin[i] += self.learning_rate * tree_pred[i]
        proba = [self._sigmoid(p) for p in pred_margin]
        return proba

    def predict(self, X, threshold=0.5):
        """Dự đoán nhãn lớp (0 hoặc 1)."""
        proba = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in proba]


# -------------------------------
# Ví dụ sử dụng (binary classification)
# -------------------------------
if __name__ == "__main__":
    # Dữ liệu mẫu: XOR-like
    X = [[0, 0], [0, 1], [1, 0], [1, 1],
         [0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]]
    y = [0, 1, 1, 0, 0, 1, 1, 0]

    # Huấn luyện XGBoost
    model = XGBoostClassifier(n_estimators=10, learning_rate=0.3, max_depth=2,
                              lambda_reg=0.1, gamma=0.0, subsample=1.0)
    model.fit(X, y)

    # Dự đoán
    preds = model.predict(X)
    probas = model.predict_proba(X)

    print("Nhãn thực tế:", y)
    print("Dự đoán     :", preds)
    print("Xác suất lớp 1:", [round(p, 4) for p in probas])