import math
import random

class LSTMCell:
    """
    Một ô LSTM đơn lẻ (cho một bước thời gian).
    Quản lý trạng thái ẩn h và trạng thái nhớ c.
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Khởi tạo ma trận trọng số và bias ngẫu nhiên nhỏ
        # Các cổng: forget (f), input (i), candidate (g), output (o)
        # W_x: input_size x hidden_size, W_h: hidden_size x hidden_size, bias: hidden_size
        def init_matrix(rows, cols):
            return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

        self.Wx_f = init_matrix(input_size, hidden_size)
        self.Wh_f = init_matrix(hidden_size, hidden_size)
        self.b_f = [0.0] * hidden_size

        self.Wx_i = init_matrix(input_size, hidden_size)
        self.Wh_i = init_matrix(hidden_size, hidden_size)
        self.b_i = [0.0] * hidden_size

        self.Wx_g = init_matrix(input_size, hidden_size)
        self.Wh_g = init_matrix(hidden_size, hidden_size)
        self.b_g = [0.0] * hidden_size

        self.Wx_o = init_matrix(input_size, hidden_size)
        self.Wh_o = init_matrix(hidden_size, hidden_size)
        self.b_o = [0.0] * hidden_size

        # Trạng thái nội bộ
        self.h = None   # hidden state
        self.c = None   # cell state

    def _sigmoid(self, x):
        """Hàm sigmoid (cho cổng)"""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)

    def _tanh(self, x):
        """Hàm tanh (cho candidate và output)"""
        return math.tanh(x)

    def _linear(self, x, W, b):
        """Tính W^T * x + b (x là vector)"""
        out = [b[j] for j in range(len(b))]
        for i in range(len(x)):
            for j in range(len(W[0])):   # W[i][j]
                out[j] += x[i] * W[i][j]
        return out

    def forward(self, x_t):
        """
        Một bước forward của LSTM.
        x_t: vector đầu vào tại thời điểm t (list chiều dài input_size)
        Trả về h_t (hidden state mới)
        """
        if self.h is None:
            # Khởi tạo trạng thái ban đầu (toàn 0)
            self.h = [0.0] * self.hidden_size
            self.c = [0.0] * self.hidden_size

        # Tính các cổng
        f_gate = self._linear(x_t, self.Wx_f, self.b_f)
        f_gate = [self._sigmoid(f_gate[j] + sum(self.Wh_f[k][j] * self.h[k] for k in range(self.hidden_size)))
                  for j in range(self.hidden_size)]

        i_gate = self._linear(x_t, self.Wx_i, self.b_i)
        i_gate = [self._sigmoid(i_gate[j] + sum(self.Wh_i[k][j] * self.h[k] for k in range(self.hidden_size)))
                  for j in range(self.hidden_size)]

        g_gate = self._linear(x_t, self.Wx_g, self.b_g)
        g_gate = [self._tanh(g_gate[j] + sum(self.Wh_g[k][j] * self.h[k] for k in range(self.hidden_size)))
                  for j in range(self.hidden_size)]

        o_gate = self._linear(x_t, self.Wx_o, self.b_o)
        o_gate = [self._sigmoid(o_gate[j] + sum(self.Wh_o[k][j] * self.h[k] for k in range(self.hidden_size)))
                  for j in range(self.hidden_size)]

        # Cập nhật cell state
        self.c = [f_gate[j] * self.c[j] + i_gate[j] * g_gate[j] for j in range(self.hidden_size)]

        # Cập nhật hidden state
        self.h = [o_gate[j] * self._tanh(self.c[j]) for j in range(self.hidden_size)]

        return self.h

    def reset_state(self):
        """Đặt lại trạng thái (h, c) về 0"""
        self.h = None
        self.c = None


class LSTM:
    """
    LSTM cho toàn bộ chuỗi (nhiều bước thời gian).
    Sử dụng một LSTMCell bên trong.
    """
    def __init__(self, input_size, hidden_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, inputs):
        """
        inputs: list các vector đầu vào theo thời gian, mỗi vector là list chiều dài input_size.
        Trả về list các hidden state (mỗi hidden state là list chiều dài hidden_size).
        """
        outputs = []
        self.cell.reset_state()
        for x_t in inputs:
            h_t = self.cell.forward(x_t)
            outputs.append(h_t)
        return outputs

    def reset_state(self):
        self.cell.reset_state()


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Cấu hình
    input_size = 3
    hidden_size = 4
    seq_len = 5

    # Dữ liệu giả: chuỗi các vector ngẫu nhiên
    inputs = []
    for _ in range(seq_len):
        vec = [random.uniform(-1, 1) for _ in range(input_size)]
        inputs.append(vec)

    print("Chuỗi đầu vào (5 bước, mỗi bước vector 3 chiều):")
    for i, v in enumerate(inputs):
        print(f"t={i+1}: {[round(x, 3) for x in v]}")

    # Tạo LSTM
    lstm = LSTM(input_size, hidden_size)

    # Forward
    outputs = lstm.forward(inputs)

    print("\nĐầu ra hidden state (mỗi bước, 4 chiều):")
    for i, h in enumerate(outputs):
        print(f"t={i+1}: {[round(x, 3) for x in h]}")