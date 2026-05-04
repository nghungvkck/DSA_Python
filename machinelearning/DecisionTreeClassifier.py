import math
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        """
        criterion: 'gini' hoặc 'entropy'
        max_depth: độ sâu tối đa của cây (None = không giới hạn)
        min_samples_split: số mẫu tối thiểu để split một node
        min_impurity_decrease: giảm impurity tối thiểu để chấp nhận split
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None  # sẽ là một dict hoặc Node

    def _impurity(self, y):
        """Tính impurity của một tập nhãn y."""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        if self.criterion == 'gini':
            # Gini = 1 - sum(p^2)
            return 1 - sum(p**2 for p in probs)
        else:  # entropy
            # Entropy = -sum(p * log2(p))
            return -sum(p * math.log2(p) for p in probs if p > 0)

    def _impurity_gain(self, y, y_left, y_right):
        """Tính độ giảm impurity khi split."""
        impurity_parent = self._impurity(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        impurity_children = (n_left / n) * self._impurity(y_left) + (n_right / n) * self._impurity(y_right)
        return impurity_parent - impurity_children

    def _best_split(self, X, y):
        """Tìm split tốt nhất trên tất cả features và ngưỡng."""
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
            # Thử split tại trung điểm giữa các giá trị liên tiếp
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
        """Xây dựng cây đệ quy."""
        n_samples = len(y)
        n_classes = len(set(y))

        # Điều kiện dừng
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split or n_classes == 1:
            # Tạo lá: lớp chiếm đa số
            most_common = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'class': most_common, 'counts': dict(Counter(y))}

        split = self._best_split(X, y)
        if split is None:
            most_common = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'class': most_common, 'counts': dict(Counter(y))}

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
            'impurity_gain': self._impurity_gain(y, y_left, y_right)
        }

    def fit(self, X, y):
        """Huấn luyện cây."""
        if len(X) != len(y):
            raise ValueError("Số lượng mẫu và nhãn không khớp.")
        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, node, x):
        """Dự đoán một mẫu."""
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        else:
            return self._predict_one(node['right'], x)

    def predict(self, X):
        """Dự đoán cho nhiều mẫu."""
        return [self._predict_one(self.tree, x) for x in X]

    def _print_tree(self, node, indent=""):
        """In cây ra màn hình (đệ quy)."""
        if node['leaf']:
            print(f"{indent}-> Dự đoán lớp {node['class']} (các mẫu: {node['counts']})")
            return
        print(f"{indent}Nếu feature {node['feature']} <= {node['threshold']:.4f}:")
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
    # Dữ liệu mẫu: XOR-like
    X = [[0, 0], [0, 1], [1, 0], [1, 1],
         [0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]]
    y = [0, 1, 1, 0, 0, 1, 1, 0]

    # Tạo cây với Gini, độ sâu tối đa 3
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=2)
    clf.fit(X, y)

    print("Cây quyết định vừa huấn luyện:")
    clf.print_tree()

    print("\nDự đoán trên tập huấn luyện:")
    preds = clf.predict(X)
    print("Thực tế:", y)
    print("Dự đoán:", preds)

    # Dự đoán mẫu mới
    X_new = [[0.1, 0.9], [0.9, 0.1]]
    print("\nDự đoán mẫu mới:", clf.predict(X_new))