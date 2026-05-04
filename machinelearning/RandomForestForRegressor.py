import math
import random
from collections import Counter

# -------------------------------------------------
# Decision Tree Regressor (hỗ trợ random feature subset)
# -------------------------------------------------
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.tree = None
        self.n_features = None

    def _variance(self, y):
        """Phương sai (tổng thể)"""
        if len(y) == 0:
            return 0.0
        mean = sum(y) / len(y)
        var = sum((v - mean) ** 2 for v in y) / len(y)
        return var

    def _mse_reduction(self, y, y_left, y_right):
        """Độ giảm MSE = var_parent - (n_left/n)*var_left - (n_right/n)*var_right"""
        var_parent = self._variance(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        if n_left == 0 or n_right == 0:
            return 0.0
        weighted_var_children = (n_left / n) * self._variance(y_left) + (n_right / n) * self._variance(y_right)
        return var_parent - weighted_var_children

    def _best_split(self, X, y, feature_indices):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None
        n_samples = len(X)

        for feat in feature_indices:
            values = sorted(set(X[i][feat] for i in range(n_samples)))
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
        n_samples = len(y)
        leaf_value = sum(y) / n_samples if n_samples > 0 else 0.0

        # Điều kiện dừng
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return {'leaf': True, 'value': leaf_value, 'n_samples': n_samples}

        # Chọn ngẫu nhiên tập feature
        if self.max_features is None:
            feature_indices = list(range(self.n_features))
        else:
            feature_indices = random.sample(range(self.n_features), self.max_features)

        split = self._best_split(X, y, feature_indices)
        if split is None:
            return {'leaf': True, 'value': leaf_value, 'n_samples': n_samples}

        feature, threshold, left_idx, right_idx = split
        X_left = [X[i] for i in left_idx]
        y_left = [y[i] for i in left_idx]
        X_right = [X[i] for i in right_idx]
        y_right = [y[i] for i in right_idx]

        left_subtree = self._build_tree(X_left, y_left, depth+1)
        right_subtree = self._build_tree(X_right, y_right, depth+1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        self.n_features = len(X[0]) if X else 0
        if self.max_features is None:
            self.max_features = self.n_features  # default: all features (or could be n_features//3)
        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, node, x):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        else:
            return self._predict_one(node['right'], x)

    def predict(self, X):
        return [self._predict_one(self.tree, x) for x in X]


# -------------------------------------------------
# Random Forest Regressor
# -------------------------------------------------
class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0, max_features=None,
                 bootstrap=True, max_samples=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features  # nếu None, mỗi cây dùng tất cả (hoặc có thể tự động)
        self.bootstrap = bootstrap
        self.max_samples = max_samples    # kích thước bootstrap sample
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n = len(X)
        sample_size = self.max_samples if self.max_samples is not None else n
        indices = random.choices(range(n), k=sample_size)  # có hoàn lại
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        return X_sample, y_sample

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Dự đoán từ tất cả các cây
        all_preds = [tree.predict(X) for tree in self.trees]
        # Trung bình theo từng mẫu
        n_samples = len(X)
        final_preds = []
        for i in range(n_samples):
            pred_i = [all_preds[t][i] for t in range(self.n_estimators)]
            final_preds.append(sum(pred_i) / self.n_estimators)
        return final_preds


# -------------------------------------------------
# Ví dụ sử dụng
# -------------------------------------------------
if __name__ == "__main__":
    # Dữ liệu mẫu: y = x1 + x2 + nhiễu nhẹ
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
    y = [3, 5, 7, 9, 11, 13, 15, 17]  # x1 + x2
    # Thêm nhiễu để không hoàn hảo
    y = [val + 0.3 * (i % 2) for i, val in enumerate(y)]

    # Huấn luyện Random Forest hồi quy
    rf_reg = RandomForestRegressor(n_estimators=10, max_depth=3,
                                   min_samples_split=2, max_features=2,
                                   bootstrap=True)
    rf_reg.fit(X, y)

    # Dự đoán trên tập huấn luyện
    preds = rf_reg.predict(X)
    print("Thực tế   :", [round(v, 2) for v in y])
    print("RF dự đoán:", [round(v, 2) for v in preds])

    # Dự đoán mẫu mới
    X_new = [[2, 2], [5, 5]]
    print("\nMẫu mới dự đoán:", [round(v, 2) for v in rf_reg.predict(X_new)])