import math
import random
from collections import defaultdict

# ------------------------------------------------------------
# Helper: Sigmoid
# ------------------------------------------------------------
def sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# ------------------------------------------------------------
# Histogram-based binning (equal‑width)
# ------------------------------------------------------------
def create_bins(X, num_bins=10):
    """
    X: list of lists, mỗi dòng là một mẫu.
    Trả về: bins = list of (bin_edges, num_bins) cho mỗi feature.
    """
    n_features = len(X[0])
    bins = []
    for f in range(n_features):
        values = [x[f] for x in X]
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            # feature hằng số
            bins.append(([min_val], 1))
            continue
        bin_width = (max_val - min_val) / num_bins
        edges = [min_val + i * bin_width for i in range(num_bins + 1)]
        edges[-1] = max_val  # đảm bảo cận trên đúng
        bins.append((edges, num_bins))
    return bins

def apply_bins(X, bins):
    """Chuyển đổi X thành các chỉ số bin (0..num_bins-1)."""
    X_binned = []
    for x in X:
        row = []
        for f, (edges, nb) in enumerate(bins):
            val = x[f]
            # tìm bin chứa val
            bin_idx = nb - 1  # default bin cuối
            for i in range(len(edges)-1):
                if edges[i] <= val <= edges[i+1]:
                    bin_idx = i
                    break
            row.append(bin_idx)
        X_binned.append(row)
    return X_binned

# ------------------------------------------------------------
# LightGBM Tree (leaf‑wise, histogram‑based)
# ------------------------------------------------------------
class LightGBMTree:
    def __init__(self, max_leaves=31, min_child_samples=20, lambda_l1=0.0, lambda_l2=0.0,
                 min_split_gain=0.0, num_bins=10):
        self.max_leaves = max_leaves          # số lá tối đa
        self.min_child_samples = min_child_samples
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_split_gain = min_split_gain
        self.num_bins = num_bins
        self.bins = None                      # sẽ gán sau khi fit
        self.nodes = []                       # lưu các node dạng dict

    def _compute_gain(self, G, H):
        """Tính gain = (G^2)/(H+lambda) với regularization L2, L1 (L1 dùng để làm sparse)."""
        # L1: |G| - lambda_l1, nhưng đơn giản chỉ trừ lambda_l1 * |G|
        # Thực tế LightGBM dùng: gain = (G^2)/(H+lambda) nếu |G| > lambda_l1, else 0
        if abs(G) <= self.lambda_l1:
            return 0.0
        G_abs = abs(G) - self.lambda_l1
        # Sau khi trừ L1, giữ nguyên dấu
        G_adj = G_abs if G > 0 else -G_abs
        return (G_adj * G_adj) / (H + self.lambda_l2)

    def _build_histograms(self, X_binned, grad, hess):
        """Xây dựng histogram cho mỗi feature: mảng sum_grad[b], sum_hess[b]."""
        n_samples = len(X_binned)
        n_features = len(X_binned[0])
        histograms = []
        for f in range(n_features):
            sum_grad = [0.0] * self.num_bins
            sum_hess = [0.0] * self.num_bins
            for i in range(n_samples):
                b = X_binned[i][f]
                sum_grad[b] += grad[i]
                sum_hess[b] += hess[i]
            histograms.append((sum_grad, sum_hess))
        return histograms

    def _find_best_split(self, histograms, node_indices, grad, hess, X_binned):
        """Tìm split tốt nhất từ histogram."""
        best_gain = -float('inf')
        best_feature = None
        best_bin = None
        best_left_mask = None

        G_total = sum(grad[i] for i in node_indices)
        H_total = sum(hess[i] for i in node_indices)
        gain_base = self._compute_gain(G_total, H_total)

        for f, (sum_grad, sum_hess) in enumerate(histograms):
            G_left = 0.0
            H_left = 0.0
            # Duyệt lần lượt các bin từ trái sang phải
            for b in range(self.num_bins - 1):
                G_left += sum_grad[b]
                H_left += sum_hess[b]
                G_right = G_total - G_left
                H_right = H_total - H_left

                # Kiểm tra số lượng mẫu mỗi nhánh
                left_cnt = sum(1 for i in node_indices if X_binned[i][f] <= b)
                right_cnt = len(node_indices) - left_cnt
                if left_cnt < self.min_child_samples or right_cnt < self.min_child_samples:
                    continue

                gain = 0.5 * (self._compute_gain(G_left, H_left) +
                              self._compute_gain(G_right, H_right) - gain_base)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_bin = b
        if best_gain < self.min_split_gain:
            return None
        # Tạo mask cho node con trái (<= bin)
        left_mask = [i for i in node_indices if X_binned[i][best_feature] <= best_bin]
        right_mask = [i for i in node_indices if i not in left_mask]
        return {'feature': best_feature, 'bin_threshold': best_bin,
                'left_indices': left_mask, 'right_indices': right_mask,
                'gain': best_gain}

    def fit(self, X, grad, hess):
        """
        X: dữ liệu gốc (list of lists) – sẽ được binning trong quá trình fit.
        grad, hess: gradient và hessian (list).
        """
        # Tạo bins và chuyển đổi X
        self.bins = create_bins(X, num_bins=self.num_bins)
        X_binned = apply_bins(X, self.bins)

        # Node gốc: chứa tất cả chỉ số mẫu
        root_indices = list(range(len(X)))
        # Giá trị dự đoán của lá = -sum(grad)/sum(hess)
        root_value = -sum(grad) / (sum(hess) + self.lambda_l2)

        leaf_queue = [(root_indices, root_value)]   # mỗi phần tử: (indices, leaf_value)
        self.nodes = [{'leaf': True, 'value': root_value, 'indices': root_indices}]

        n_leaves = 1
        while n_leaves < self.max_leaves:
            # Tìm lá có gain lớn nhất nếu split
            best_leaf_idx = -1
            best_split_info = None
            best_leaf_gain = -float('inf')

            for leaf_idx, node in enumerate(self.nodes):
                if not node['leaf']:
                    continue
                indices = node['indices']
                if len(indices) < 2 * self.min_child_samples:
                    continue
                # Xây histogram chỉ cho các mẫu trong lá này
                g_leaf = [grad[i] for i in indices]
                h_leaf = [hess[i] for i in indices]
                X_leaf_binned = [X_binned[i] for i in indices]

                # Histogram cho từng feature
                hist = self._build_histograms(X_leaf_binned, g_leaf, h_leaf)
                split = self._find_best_split(hist, list(range(len(indices))),
                                              g_leaf, h_leaf, X_leaf_binned)
                if split is not None and split['gain'] > best_leaf_gain:
                    best_leaf_gain = split['gain']
                    best_leaf_idx = leaf_idx
                    # Lưu lại split với indices thật (toàn bộ dữ liệu)
                    left_real = [indices[i] for i in split['left_indices']]
                    right_real = [indices[i] for i in split['right_indices']]
                    best_split_info = {
                        'feature': split['feature'],
                        'bin': split['bin_threshold'],
                        'left_indices': left_real,
                        'right_indices': right_real
                    }

            if best_leaf_idx == -1:
                break  # không thể split thêm

            # Thực hiện split lá tốt nhất
            node = self.nodes[best_leaf_idx]
            node['leaf'] = False
            node['split_feature'] = best_split_info['feature']
            node['split_bin'] = best_split_info['bin']
            # Tạo node con trái
            left_indices = best_split_info['left_indices']
            left_value = -sum(grad[i] for i in left_indices) / (sum(hess[i] for i in left_indices) + self.lambda_l2)
            # Tạo node con phải
            right_indices = best_split_info['right_indices']
            right_value = -sum(grad[i] for i in right_indices) / (sum(hess[i] for i in right_indices) + self.lambda_l2)

            left_node = {'leaf': True, 'value': left_value, 'indices': left_indices}
            right_node = {'leaf': True, 'value': right_value, 'indices': right_indices}
            node['left'] = len(self.nodes)
            node['right'] = len(self.nodes) + 1
            self.nodes.append(left_node)
            self.nodes.append(right_node)
            n_leaves += 1

        # Xóa trường 'indices' để tiết kiệm bộ nhớ (không cần sau khi train)
        for node in self.nodes:
            if 'indices' in node:
                del node['indices']
        return self

    def predict_one(self, node_idx, x):
        """Dự đoán một mẫu x (giá trị gốc, chưa bin) dựa vào node hiện tại."""
        node = self.nodes[node_idx]
        if node['leaf']:
            return node['value']
        # Bin giá trị x tại feature split
        f = node['split_feature']
        bin_edges, _ = self.bins[f]
        # tìm bin của x[f]
        val = x[f]
        for i in range(len(bin_edges)-1):
            if bin_edges[i] <= val <= bin_edges[i+1]:
                bin_idx = i
                break
        else:
            bin_idx = len(bin_edges)-2
        if bin_idx <= node['split_bin']:
            return self.predict_one(node['left'], x)
        else:
            return self.predict_one(node['right'], x)

    def predict(self, X):
        return [self.predict_one(0, x) for x in X]

# ------------------------------------------------------------
# LightGBM Classifier (Gradient Boosting Framework)
# ------------------------------------------------------------
class LightGBMClassifier:
    def __init__(self, num_iterations=100, learning_rate=0.1, max_leaves=31,
                 min_child_samples=20, lambda_l1=0.0, lambda_l2=0.0,
                 min_split_gain=0.0, num_bins=10):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.min_child_samples = min_child_samples
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_split_gain = min_split_gain
        self.num_bins = num_bins
        self.trees = []
        self.initial_pred = None

    def _logistic_loss_grad_hess(self, y_true, y_pred_prob):
        """y_true: 0/1, y_pred_prob: xác suất dự đoán."""
        grad = [p - y for p, y in zip(y_pred_prob, y_true)]
        hess = [p * (1.0 - p) for p in y_pred_prob]
        return grad, hess

    def fit(self, X, y):
        n_samples = len(X)
        # Khởi tạo dự đoán log‑odds
        mean_y = sum(y) / n_samples
        self.initial_pred = math.log(mean_y / (1.0 - mean_y + 1e-15))
        pred_margin = [self.initial_pred] * n_samples

        for iteration in range(self.num_iterations):
            pred_prob = [sigmoid(p) for p in pred_margin]
            grad, hess = self._logistic_loss_grad_hess(y, pred_prob)

            # Xây dựng cây Leaf‑wise
            tree = LightGBMTree(max_leaves=self.max_leaves,
                                min_child_samples=self.min_child_samples,
                                lambda_l1=self.lambda_l1,
                                lambda_l2=self.lambda_l2,
                                min_split_gain=self.min_split_gain,
                                num_bins=self.num_bins)
            tree.fit(X, grad, hess)
            tree_pred = tree.predict(X)
            # Cập nhật margin
            for i in range(n_samples):
                pred_margin[i] += self.learning_rate * tree_pred[i]
            self.trees.append(tree)

    def predict_proba(self, X):
        pred_margin = [self.initial_pred] * len(X)
        for tree in self.trees:
            tree_pred = tree.predict(X)
            for i in range(len(X)):
                pred_margin[i] += self.learning_rate * tree_pred[i]
        return [sigmoid(p) for p in pred_margin]

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in proba]

# ------------------------------------------------------------
# Ví dụ sử dụng
# ------------------------------------------------------------
if __name__ == "__main__":
    # Tạo dữ liệu giả: XOR-like
    X = [[0, 0], [0, 1], [1, 0], [1, 1],
         [0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]]
    y = [0, 1, 1, 0, 0, 1, 1, 0]

    # Huấn luyện LightGBM
    model = LightGBMClassifier(num_iterations=10, learning_rate=0.2,
                               max_leaves=5, min_child_samples=2,
                               lambda_l2=0.1, num_bins=5)
    model.fit(X, y)

    # Dự đoán
    preds = model.predict(X)
    probas = model.predict_proba(X)
    print("Thực tế   :", y)
    print("Dự đoán   :", preds)
    print("Xác suất P1:", [round(p, 3) for p in probas])

    # Dự đoán mẫu mới
    X_new = [[0.1, 0.9], [0.9, 0.1]]
    print("\nMẫu mới dự đoán:", model.predict(X_new))