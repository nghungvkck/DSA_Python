import math
import random

class CNN1D:
    """
    Lớp Conv1D đơn giản cho NLP: nhận đầu vào là chuỗi các embedding vector (seq_len x embed_dim)
    Áp dụng các bộ lọc (kernel_size x embed_dim), tạo ra feature maps, pooling, và fully connected.
    """
    def __init__(self, embed_dim, num_filters, kernel_sizes, activation='relu', pool='max'):
        """
        embed_dim: số chiều của embedding vector đầu vào
        num_filters: số lượng filter cho mỗi kernel size (có thể là int hoặc list cùng độ dài kernel_sizes)
        kernel_sizes: list các kích thước kernel (ví dụ [2,3,4])
        activation: 'relu', 'tanh', 'sigmoid'
        pool: 'max' hoặc 'avg'
        """
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        if isinstance(num_filters, int):
            self.num_filters = [num_filters] * len(kernel_sizes)
        else:
            self.num_filters = num_filters
        self.activation = activation
        self.pool = pool
        
        # Khởi tạo bộ lọc và bias cho từng kernel size
        self.convs = []  # mỗi phần tử là list các filter, mỗi filter là ma trận (kernel_size x embed_dim)
        self.biases = []
        for idx, k in enumerate(kernel_sizes):
            filters = []
            for _ in range(self.num_filters[idx]):
                # kernel shape: (k, embed_dim)
                kernel = [[random.uniform(-0.1, 0.1) for _ in range(embed_dim)] for _ in range(k)]
                filters.append(kernel)
            self.convs.append(filters)
            self.biases.append([0.0] * self.num_filters[idx])
    
    def _activation_func(self, x):
        if self.activation == 'relu':
            return max(0, x)
        elif self.activation == 'tanh':
            return math.tanh(x)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + math.exp(-x))
        else:
            return x
    
    def _conv1d_single_filter(self, emb_seq, kernel, bias, kernel_size):
        """
        Tính convolution cho một filter trên toàn bộ chuỗi.
        emb_seq: list of list (seq_len x embed_dim)
        kernel: list of list (kernel_size x embed_dim)
        bias: float
        kernel_size: int
        Trả về list output (độ dài seq_len - kernel_size + 1)
        """
        seq_len = len(emb_seq)
        out_len = seq_len - kernel_size + 1
        out = []
        for i in range(out_len):
            window = emb_seq[i:i+kernel_size]
            total = bias
            for h in range(kernel_size):
                for w in range(self.embed_dim):
                    total += window[h][w] * kernel[h][w]
            out.append(self._activation_func(total))
        return out
    
    def forward(self, emb_seq):
        """
        emb_seq: list of list (seq_len x embed_dim)
        Trả về vector đặc trưng sau khi concat các pooling results (độ dài = tổng num_filters)
        """
        all_features = []
        for idx, k in enumerate(self.kernel_sizes):
            filters = self.convs[idx]
            biases = self.biases[idx]
            feature_maps = []
            for f_idx in range(self.num_filters[idx]):
                conv_out = self._conv1d_single_filter(emb_seq, filters[f_idx], biases[f_idx], k)
                # Pooling
                if self.pool == 'max':
                    pooled = max(conv_out) if conv_out else 0.0
                else:  # avg
                    pooled = sum(conv_out) / len(conv_out) if conv_out else 0.0
                feature_maps.append(pooled)
            all_features.extend(feature_maps)
        return all_features  # vector đặc trưng


class CNNTextClassifier:
    """
    Mô hình CNN hoàn chỉnh phân loại văn bản (hoặc biểu diễn câu).
    Đầu vào: các embedding vector (đã có sẵn, không cần embedding layer)
    """
    def __init__(self, embed_dim, num_filters, kernel_sizes, num_classes, pool='max', activation='relu'):
        self.cnn = CNN1D(embed_dim, num_filters, kernel_sizes, activation, pool)
        # Tính số chiều đầu vào cho FC
        if isinstance(num_filters, int):
            fc_in = len(kernel_sizes) * num_filters
        else:
            fc_in = sum(num_filters)
        self.fc_weight = [[random.uniform(-0.1, 0.1) for _ in range(fc_in)] for _ in range(num_classes)]
        self.fc_bias = [0.0] * num_classes
        self.num_classes = num_classes
    
    def _softmax(self, logits):
        max_val = max(logits)
        exp = [math.exp(l - max_val) for l in logits]
        s = sum(exp)
        return [e / s for e in exp]
    
    def forward(self, emb_seq):
        """
        emb_seq: list of list (seq_len x embed_dim)
        Trả về vector xác suất (list, num_classes)
        """
        features = self.cnn.forward(emb_seq)  # vector
        logits = [self.fc_bias[i] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            total = self.fc_bias[i]
            for j in range(len(features)):
                total += self.fc_weight[i][j] * features[j]
            logits[i] = total
        probs = self._softmax(logits)
        return probs
    
    def predict(self, emb_seq):
        probs = self.forward(emb_seq)
        return max(range(self.num_classes), key=lambda i: probs[i])


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Giả sử chúng ta có các vector embedding cho mỗi từ (ví dụ: từ 5 chiều)
    # Tạo embedding giả: câu "I like NLP" có 3 token, mỗi token vector 4 chiều
    embed_dim = 4
    seq_len = 3
    # Tạo embedding ngẫu nhiên cho câu
    emb_seq = [[random.uniform(-1, 1) for _ in range(embed_dim)] for _ in range(seq_len)]
    
    print("Embedding đầu vào (seq_len=3, embed_dim=4):")
    for i, v in enumerate(emb_seq):
        print(f"Token {i+1}: {[round(x,2) for x in v]}")
    
    # Tạo CNN classifier 2 class
    classifier = CNNTextClassifier(embed_dim=embed_dim, 
                                   num_filters=3, 
                                   kernel_sizes=[2,3], 
                                   num_classes=2,
                                   pool='max')
    
    probs = classifier.forward(emb_seq)
    print("\nXác suất:", [round(p,3) for p in probs])
    pred = classifier.predict(emb_seq)
    print("Dự đoán class:", pred)