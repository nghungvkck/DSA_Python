# -------------------------------------------------
# Decision Tree (hỗ trợ random feature subset)
# -------------------------------------------------
import math
import random
from collections import Counter

# -------------------------------------------------
# Decision Tree (hỗ trợ random feature subset)
# -------------------------------------------------
class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0, max_features=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features  # số lượng feature xét tại mỗi split
        self.tree = None
        self.n_features = None

    def _impurity(self, y):
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probs = [cnt / len(y) for cnt in counts.values()]
        if self.criterion == 'gini':
            return 1 - sum(p**2 for p in probs)
        else:  # entropy
            return -sum(p * math.log2(p) for p in probs if p > 0)

    def _impurity_gain(self, y, y_left, y_right):
        imp_parent = self._impurity(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        imp_child = (n_left / n) * self._impurity(y_left) + (n_right / n) * self._impurity(y_right)
        return imp_parent - imp_child

    def _best_split(self, X, y, feature_indices):
        """Tìm split tốt nhất chỉ trên tập feature_indices."""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None
        n_samples = len(X)

        for feat in feature_indices:
            # Lấy các giá trị feature khác nhau
            values = sorted(set(X[i][feat] for i in range(n_samples)))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i+1]) / 2.0

                left_idx = [idx for idx in range(n_samples) if X[idx][feat] <= threshold]
                right_idx = [idx for idx in range(n_samples) if X[idx][feat] > threshold]

                if len(left_idx) < self.min_samples_split or len(right_idx) < self.min_samples_split:
                    continue

                y_left = [y[idx] for idx in left_idx]
                y_right = [y[idx] for idx in right_idx]
                gain = self._impurity_gain(y, y_left, y_right)

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
        n_classes = len(set(y))
        leaf_value = Counter(y).most_common(1)[0][0]

        # Điều kiện dừng
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split or n_classes == 1:
            return {'leaf': True, 'class': leaf_value, 'n_samples': n_samples}

        # Chọn ngẫu nhiên tập feature để xét (feature bagging)
        if self.max_features is None:
            feature_indices = list(range(self.n_features))
        else:
            feature_indices = random.sample(range(self.n_features), self.max_features)

        split = self._best_split(X, y, feature_indices)
        if split is None:
            return {'leaf': True, 'class': leaf_value, 'n_samples': n_samples}

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

        # ✅ FIX: xử lý max_features dạng string giống sklearn
        if self.max_features is None:
            self.max_features = int(math.sqrt(self.n_features))
        elif self.max_features == 'sqrt':
            self.max_features = int(math.sqrt(self.n_features))
        elif self.max_features == 'log2':
            self.max_features = int(math.log2(self.n_features))
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * self.n_features)

        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, node, x):
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        else:
            return self._predict_one(node['right'], x)

    def predict(self, X):
        return [self._predict_one(self.tree, x) for x in X]


# -------------------------------------------------
# Random Forest Classifier
# -------------------------------------------------
class RandomForestClassifier:
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
                 min_samples_split=2, min_impurity_decrease=0.0, max_features=None,
                 bootstrap=True, max_samples=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features  # nếu None thì tự động sqrt(n_features)
        self.bootstrap = bootstrap
        self.max_samples = max_samples  # kích thước bootstrap sample (mặc định = n_samples)
        self.trees = []
        self.classes_ = None  # ✅ FIX: lưu danh sách class

    def _bootstrap_sample(self, X, y):
        n = len(X)
        sample_size = self.max_samples if self.max_samples is not None else n
        indices = random.choices(range(n), k=sample_size)  # lấy mẫu có hoàn lại
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        return X_sample, y_sample

    def fit(self, X, y):
        self.trees = []
        self.classes_ = sorted(set(y))  # ✅ FIX

        for _ in range(self.n_estimators):
            # Lấy bootstrap sample
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            # Tạo cây với tham số max_features
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Dự đoán từ tất cả các cây
        all_preds = []
        for tree in self.trees:
            all_preds.append(tree.predict(X))

        n_samples = len(X)
        final_preds = []
        for i in range(n_samples):
            votes = [all_preds[t][i] for t in range(len(self.trees))]
            majority_class = Counter(votes).most_common(1)[0][0]
            final_preds.append(majority_class)

        return final_preds

    def predict_proba(self, X):
        """Trả về xác suất trung bình của các cây (tần suất dự đoán)."""
        all_preds = []
        for tree in self.trees:
            all_preds.append(tree.predict(X))

        n_samples = len(X)
        probas = []

        for i in range(n_samples):
            votes = [all_preds[t][i] for t in range(len(self.trees))]

            # ✅ FIX: dùng class cố định
            prob = [votes.count(c) / len(votes) for c in self.classes_]

            probas.append(prob)

        return probas


# -------------------------------------------------
# Ví dụ sử dụng
# -------------------------------------------------
if __name__ == "__main__":
    # Dữ liệu mẫu: XOR-like
    X = [[0, 0], [0, 1], [1, 0], [1, 1],
         [0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]]
    y = [0, 1, 1, 0, 0, 1, 1, 0]

    # Huấn luyện Random Forest
    rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=3,
                                min_samples_split=2, max_features='sqrt')
    rf.fit(X, y)

    # Dự đoán
    preds = rf.predict(X)
    print("Thực tế:", y)
    print("RF dự đoán:", preds)

    # Dự đoán mẫu mới
    X_new = [[0.1, 0.9], [0.9, 0.1]]
    print("\nMẫu mới dự đoán:", rf.predict(X_new))

    # Xác suất (nếu cần)
    probas = rf.predict_proba(X_new)
    print("Xác suất mỗi lớp:", probas)