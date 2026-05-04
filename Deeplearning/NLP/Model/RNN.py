import math
import random

# ===== ACTIVATION =====
def tanh(x):
    return math.tanh(x)

def dtanh(x):
    return 1 - math.tanh(x) ** 2   # đạo hàm tanh

# ===== SOFTMAX + LOSS =====
def softmax(logits):
    m = max(logits)  # tránh overflow
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def cross_entropy(pred, target):
    return -math.log(pred[target] + 1e-9)  # tránh log(0)

# ===== MODEL =====
class RNNClassifier:
    def __init__(self, input_size, hidden_size, num_classes, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lr = lr

        # W_xh: input → hidden
        self.W_xh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(input_size)]
        # W_hh: hidden → hidden (quan trọng của RNN)
        self.W_hh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b_h = [0.0] * hidden_size

        # FC: hidden → output
        self.W_hy = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(num_classes)]
        self.b_y = [0.0] * num_classes

    # ===== FORWARD =====
    def forward(self, seq):
        self.xs = []   # lưu input từng bước
        self.hs = []   # lưu hidden states

        h = [0.0] * self.hidden_size
        self.hs.append(h)

        # chạy qua từng timestep
        for x in seq:
            self.xs.append(x)

            new_h = [self.b_h[j] for j in range(self.hidden_size)]

            # W_xh * x
            for i in range(self.input_size):
                for j in range(self.hidden_size):
                    new_h[j] += x[i] * self.W_xh[i][j]

            # W_hh * h_prev
            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    new_h[j] += h[i] * self.W_hh[i][j]

            # activation
            h = [tanh(v) for v in new_h]
            self.hs.append(h)

        # ===== FC layer =====
        logits = [self.b_y[i] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            for j in range(self.hidden_size):
                logits[i] += self.W_hy[i][j] * h[j]

        return softmax(logits)

    # ===== BACKWARD (BPTT) =====
    def backward(self, target):
        # gradient init
        dW_xh = [[0]*self.hidden_size for _ in range(self.input_size)]
        dW_hh = [[0]*self.hidden_size for _ in range(self.hidden_size)]
        db_h = [0]*self.hidden_size

        dW_hy = [[0]*self.hidden_size for _ in range(self.num_classes)]
        db_y = [0]*self.num_classes

        # ===== output gradient =====
        probs = self.forward(self.xs)
        probs[target] -= 1  # dL/dy

        # FC gradient
        h_last = self.hs[-1]
        for i in range(self.num_classes):
            for j in range(self.hidden_size):
                dW_hy[i][j] += probs[i] * h_last[j]
            db_y[i] += probs[i]

        # ===== BPTT =====
        dh_next = [0]*self.hidden_size

        for t in reversed(range(len(self.xs))):
            h = self.hs[t+1]
            h_prev = self.hs[t]

            # gradient từ output + future
            dh = [dh_next[j] for j in range(self.hidden_size)]
            for i in range(self.num_classes):
                for j in range(self.hidden_size):
                    dh[j] += probs[i] * self.W_hy[i][j]

            # qua tanh
            dh_raw = [dh[j] * dtanh(h[j]) for j in range(self.hidden_size)]

            # bias
            for j in range(self.hidden_size):
                db_h[j] += dh_raw[j]

            # W_xh
            for i in range(self.input_size):
                for j in range(self.hidden_size):
                    dW_xh[i][j] += self.xs[t][i] * dh_raw[j]

            # W_hh
            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    dW_hh[i][j] += h_prev[i] * dh_raw[j]

            # truyền gradient về bước trước
            dh_next = [0]*self.hidden_size
            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    dh_next[i] += dh_raw[j] * self.W_hh[i][j]

        # ===== UPDATE =====
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.W_xh[i][j] -= self.lr * dW_xh[i][j]

        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                self.W_hh[i][j] -= self.lr * dW_hh[i][j]

        for j in range(self.hidden_size):
            self.b_h[j] -= self.lr * db_h[j]

        for i in range(self.num_classes):
            for j in range(self.hidden_size):
                self.W_hy[i][j] -= self.lr * dW_hy[i][j]
            self.b_y[i] -= self.lr * db_y[i]

    # ===== TRAIN =====
    def train(self, data, labels, epochs=10):
        for e in range(epochs):
            total_loss = 0
            for x, y in zip(data, labels):
                probs = self.forward(x)
                loss = cross_entropy(probs, y)
                total_loss += loss
                self.backward(y)
            print(f"Epoch {e+1}, Loss: {round(total_loss,4)}")