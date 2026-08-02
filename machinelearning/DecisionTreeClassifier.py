import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Node:
    def __init__( self, feature = None, threshold = None, left = None, right =None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth = 10, min_sample_split = 2):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None

    def fit(self, X, y):
        X = np.array(X)  # chuyển về mảng của numpy để tận dụng cho tính toán nhanh
        y = np.array(y)
        self.root = self._build_tree(X,y, depth = 0)


    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # đoạn này có thể làm lệch dữ liệu
        if ( depth > self.max_depth or n_labels == 1 or n_samples <= self.min_sample_split):
            leaf_value = self.most_common_label(y)
            return Node(value= leaf_value)

        # Tìm đặc trưng tốt nhất, và ngưỡng chia tốt nhất
        best_feat, best_thresh = self._best_split(X,y, n_features= n_features)
        left_idx = np.where(X[:,best_feat] <= best_thresh)[0]
        right_idx = np.where(X[:,best_feat] > best_thresh)[0]

        left_child = self._build_tree(X[left_idx,:], y[left_idx] , depth+1)
        right_child = self._build_tree(X[right_idx,:], y[right_idx] , depth+1)

        return Node(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right = right_child,
        )


    # Gini
    def _best_split(self, X, y, n_features):
        best_gini = 999.0
        split_idx, split_thresh = None, None
        for feat_idx in range(n_features):
            X_columns = X[:, feat_idx]
            unique_vals = np.unique(X_columns)  # Lấy các giá trị unique
            threshold = []
            for i in range(len(unique_vals)-1):
                threshold.append((unique_vals[i] + unique_vals[i+1]) / 2)

            for thres in threshold:
                left_idx = np.where(X_columns <= thres)[0] # lấy ra các hàng thỏa mãn điều kiện
                right_idx = np.where(X_columns > thres)[0]
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                gini = self._gini(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_thresh = thres
        return split_idx, split_thresh

    def _gini(self, left_y, right_y):
        n_l, n_r = len(left_y), len(right_y)
        n_total  = n_l + n_r

        p_l = np.bincount(left_y) / n_l
        gini_l = 1.0 - np.sum(p_l ** 2)

        p_r = np.bincount(right_y) / n_r
        gini_r = 1.0 - np.sum(p_r **2 )

        weight_gini = (n_l / n_total) * gini_l + (n_r / n_total) * gini_r
        return weight_gini

    def most_common_label(self, y):
        return np.bincount(y).argmax()

        # Hàm dự đoán cho cả tập dữ liệu
    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # Hàm đệ quy duyệt cây để tìm kết quả cho 1 mẫu dữ liệu
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)



if __name__ == "__main__":
    Decisiontree = DecisionTree()
    X_train_tree = None
    Y_train_tree = None
    X_test_tree = None
    y_test_tree = None

    Decisiontree.fit(X_train_tree, Y_train_tree)
    y_pred = Decisiontree.predict(X_test_tree)
    print("Accuracy:", accuracy_score(y_test_tree, y_pred=y_pred))
