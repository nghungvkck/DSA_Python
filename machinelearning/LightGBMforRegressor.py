import math
import random

# ------------------------------------------------------------
# Helper: create bins (equal‑width)
# ------------------------------------------------------------
def create_bins(X, num_bins=10):
    """
    X: list of lists (mỗi dòng là một mẫu)
    Trả về: bins = list of (bin_edges, num_bins) cho mỗi feature.
    """
    n_features = len(X[0])
    bins = []
    for f in range(n_features):
        values = [x[f] for x in X]
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            # feature hằng số -> chỉ một bin
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
            bin_idx = nb - 1  # mặc định bin cuối
            for i in range(len(edges)-1):
                if edges[i] <= val <= edges[i+1]:
                    bin_idx = i
                    break
            row.append(bin_idx)
        X_binned.append(row)
    return X_binned

# ------------------------------------------------------------
# LightGBM Tree cho regression (leaf‑wise, histogram)
# ------------------------------------------------------------
class LightGBMRegTree:
    def __init__(self, max_leaves=31, min_child_samples=20, lambda_l1=0.0, lambda_l2=0.0,
                 min_split_gain=0.0, num_bins=10):
        self.max_leaves = max_leaves
        self.min_child_samples = min_child_samples
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_split_gain = min_split_gain
        self.num_bins = num_bins
        self.bins = None          # sẽ gán sau khi fit
        self.nodes = []           # lưu các node dạng dict

    def _compute_gain(self, G, H):
        """
        Tính gain cho regression: gain = (G^2) / (H + lambda2)
        với L1 regularization: nếu |G| <= lambda1 thì gain = 0.
        """
        if abs(G) <= self.lambda_l1:
            return 0.0
        G_abs = abs(G) - self.lambda_l1
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
        """Tìm split tốt nhất từ histogram cho node hiện tại."""
        best_gain = -float('inf')
        best_feature = None
        best_bin = None

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
        X: dữ liệu gốc (list of lists) – sẽ được binning.
        grad, hess: gradient và hessian cho từng mẫu (với regression: hess=1).
        """
        # Tạo bins và chuyển đổi X
        self.bins = create_bins(X, num_bins=self.num_bins)
        X_binned = apply_bins(X, self.bins)

        # Node gốc: chứa tất cả chỉ số mẫu
        root_indices = list(range(len(X)))
        root_value = -sum(grad) / (sum(hess) + self.lambda_l2)

        leaf_queue = [(root_indices, root_value)]   # (indices, leaf_value)
        self.nodes = [{'leaf': True, 'value': root_value, 'indices': root_indices}]

        n_leaves = 1
        while n_leaves < self.max_leaves:
            best_leaf_idx = -1
            best_split_info = None
            best_leaf_gain = -float('inf')

            for leaf_idx, node in enumerate(self.nodes):
                if not node['leaf']:
                    continue
                indices = node['indices']
                if len(indices) < 2 * self.min_child_samples:
                    continue
                # Lấy gradient, hessian của các mẫu trong lá này
                g_leaf = [grad[i] for i in indices]
                h_leaf = [hess[i] for i in indices]
                X_leaf_binned = [X_binned[i] for i in indices]

                hist = self._build_histograms(X_leaf_binned, g_leaf, h_leaf)
                split = self._find_best_split(hist, list(range(len(indices))),
                                              g_leaf, h_leaf, X_leaf_binned)
                if split is not None and split['gain'] > best_leaf_gain:
                    best_leaf_gain = split['gain']
                    best_leaf_idx = leaf_idx
                    # Chuyển indices thật ra ngoài
                    left_real = [indices[i] for i in split['left_indices']]
                    right_real = [indices[i] for i in split['right_indices']]
                    best_split_info = {
                        'feature': split['feature'],
                        'bin': split['bin_threshold'],
                        'left_indices': left_real,
                        'right_indices': right_real
                    }

            if best_leaf_idx == -1:
                break

            # Thực hiện split lá tốt nhất
            node = self.nodes[best_leaf_idx]
            node['leaf'] = False
            node['split_feature'] = best_split_info['feature']
            node['split_bin'] = best_split_info['bin']

            left_indices = best_split_info['left_indices']
            left_value = -sum(grad[i] for i in left_indices) / (sum(hess[i] for i in left_indices) + self.lambda_l2)
            right_indices = best_split_info['right_indices']
            right_value = -sum(grad[i] for i in right_indices) / (sum(hess[i] for i in right_indices) + self.lambda_l2)

            left_node = {'leaf': True, 'value': left_value, 'indices': left_indices}
            right_node = {'leaf': True, 'value': right_value, 'indices': right_indices}
            node['left'] = len(self.nodes)
            node['right'] = len(self.nodes) + 1
            self.nodes.append(left_node)
            self.nodes.append(right_node)
            n_leaves += 1

        # Xóa 'indices' để tiết kiệm bộ nhớ
        for node in self.nodes:
            if 'indices' in node:
                del node['indices']

    def _predict_one(self, node_idx, x):
        node = self.nodes[node_idx]
        if node['leaf']:
            return node['value']
        f = node['split_feature']
        # Tìm bin của x[f]
        val = x[f]
        edges, nb = self.bins[f]
        bin_idx = nb - 1
        for i in range(len(edges)-1):
            if edges[i] <= val <= edges[i+1]:
                bin_idx = i
                break
        if bin_idx <= node['split_bin']:
            return self._predict_one(node['left'], x)
        else:
            return self._predict_one(node['right'], x)

    def predict(self, X):
        return [self._predict_one(0, x) for x in X]

# ------------------------------------------------------------
# LightGBM Regressor (Gradient Boosting Framework)
# ------------------------------------------------------------
class LightGBMRegressor:
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

    def _squared_loss_grad_hess(self, y_true, y_pred):
        """Gradient = pred - y_true, Hessian = 1 (cho loss = 0.5*(y-p)^2)."""
        grad = [p - y for p, y in zip(y_pred, y_true)]
        hess = [1.0] * len(y_true)
        return grad, hess

    def fit(self, X, y):
        n_samples = len(X)
        self.initial_pred = sum(y) / n_samples
        pred = [self.initial_pred] * n_samples

        for _ in range(self.num_iterations):
            grad, hess = self._squared_loss_grad_hess(y, pred)

            # Xây dựng cây leaf‑wise histogram
            tree = LightGBMRegTree(max_leaves=self.max_leaves,
                                   min_child_samples=self.min_child_samples,
                                   lambda_l1=self.lambda_l1,
                                   lambda_l2=self.lambda_l2,
                                   min_split_gain=self.min_split_gain,
                                   num_bins=self.num_bins)
            tree.fit(X, grad, hess)
            tree_pred = tree.predict(X)

            # Cập nhật dự đoán
            for i in range(n_samples):
                pred[i] += self.learning_rate * tree_pred[i]

            self.trees.append(tree)

    def predict(self, X):
        pred = [self.initial_pred] * len(X)
        for tree in self.trees:
            tree_pred = tree.predict(X)
            for i in range(len(X)):
                pred[i] += self.learning_rate * tree_pred[i]
        return pred

# ------------------------------------------------------------
# Ví dụ sử dụng
# ------------------------------------------------------------
if __name__ == "__main__":
    # Dữ liệu mẫu: y = x1 + x2 + nhiễu nhẹ
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
    y = [3, 5, 7, 9, 11, 13, 15, 17]  # x1 + x2
    # Thêm nhiễu
    y = [val + 0.2 * (i % 2) for i, val in enumerate(y)]

    # Huấn luyện LightGBM Regressor
    model = LightGBMRegressor(num_iterations=10, learning_rate=0.15,
                              max_leaves=5, min_child_samples=2,
                              lambda_l2=0.1, num_bins=5)
    model.fit(X, y)

    # Dự đoán trên tập huấn luyện
    preds = model.predict(X)
    print("Thực tế   :", [round(v, 2) for v in y])
    print("Dự đoán   :", [round(v, 2) for v in preds])

    # Dự đoán mẫu mới
    X_new = [[2, 2], [5, 5]]
    print("Mẫu mới dự đoán:", [round(v, 2) for v in model.predict(X_new)])