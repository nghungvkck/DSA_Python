import numpy as np
from sklearn.metrics import accuracy_score

class LSTMCell:
    """Một cell LSTM đơn"""
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._init_weights()
        self.cache = {}
    
    def _init_weights(self):
        """Khởi tạo weights với Xavier initialization"""
        scale = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        
        # Forget gate
        self.W_f = np.random.randn(self.input_size, self.hidden_size) * scale
        self.U_f = np.random.randn(self.hidden_size, self.hidden_size) * scale
        self.b_f = np.zeros((1, self.hidden_size))
        
        # Input gate
        self.W_i = np.random.randn(self.input_size, self.hidden_size) * scale
        self.U_i = np.random.randn(self.hidden_size, self.hidden_size) * scale
        self.b_i = np.zeros((1, self.hidden_size))
        
        # Candidate cell
        self.W_c = np.random.randn(self.input_size, self.hidden_size) * scale
        self.U_c = np.random.randn(self.hidden_size, self.hidden_size) * scale
        self.b_c = np.zeros((1, self.hidden_size))
        
        # Output gate
        self.W_o = np.random.randn(self.input_size, self.hidden_size) * scale
        self.U_o = np.random.randn(self.hidden_size, self.hidden_size) * scale
        self.b_o = np.zeros((1, self.hidden_size))
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        return 1 - x ** 2
    
    def forward(self, x, h_prev, c_prev):
        """Forward một step"""
        f = self._sigmoid(np.dot(x, self.W_f) + np.dot(h_prev, self.U_f) + self.b_f)          # Forget gate
        i = self._sigmoid(np.dot(x, self.W_i) + np.dot(h_prev, self.U_i) + self.b_i)          # Input gate
        c_tilde = self._tanh(np.dot(x, self.W_c) + np.dot(h_prev, self.U_c) + self.b_c)       # Candidate
        c = f * c_prev + i * c_tilde                                                          # Cell state
        o = self._sigmoid(np.dot(x, self.W_o) + np.dot(h_prev, self.U_o) + self.b_o)          # Output gate
        h = o * self._tanh(c)           
        
        # Cache
        self.cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'f': f, 'i': i, 'c_tilde': c_tilde,
            'o': o, 'c': c, 'h': h
        }
        
        return h, c
    
    def backward(self, dh_next, dc_next):
        """Backward một step"""
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        c_prev = self.cache['c_prev']
        f = self.cache['f']
        i = self.cache['i']
        c_tilde = self.cache['c_tilde']
        o = self.cache['o']
        c = self.cache['c']
        h = self.cache['h']
        
        # Gradient của output gate
        do = dh_next * self._tanh(c) * self._sigmoid_derivative(o)
        
        # Gradient của cell state
        dc = dh_next * o * self._tanh_derivative(c) + dc_next
        
        # Gradient của candidate và input gate
        dc_tilde = dc * i
        di = dc * c_tilde
        
        # Gradient của forget gate
        df = dc * c_prev
        
        # Gradient của các gate (sau activation)
        df = df * self._sigmoid_derivative(f)
        di = di * self._sigmoid_derivative(i)
        dc_tilde = dc_tilde * self._tanh_derivative(c_tilde)
        do = do * self._sigmoid_derivative(o)
        
        # Gradient của weights và biases
        dW_f = np.dot(x.T, df)
        dU_f = np.dot(h_prev.T, df)
        db_f = np.sum(df, axis=0, keepdims=True)
        
        dW_i = np.dot(x.T, di)
        dU_i = np.dot(h_prev.T, di)
        db_i = np.sum(di, axis=0, keepdims=True)
        
        dW_c = np.dot(x.T, dc_tilde)
        dU_c = np.dot(h_prev.T, dc_tilde)
        db_c = np.sum(dc_tilde, axis=0, keepdims=True)
        
        dW_o = np.dot(x.T, do)
        dU_o = np.dot(h_prev.T, do)
        db_o = np.sum(do, axis=0, keepdims=True)
        
        # Gradient của input và hidden state trước
        dx = (np.dot(df, self.W_f.T) + 
              np.dot(di, self.W_i.T) + 
              np.dot(dc_tilde, self.W_c.T) + 
              np.dot(do, self.W_o.T))
        
        dh_prev = (np.dot(df, self.U_f.T) + 
                   np.dot(di, self.U_i.T) + 
                   np.dot(dc_tilde, self.U_c.T) + 
                   np.dot(do, self.U_o.T))
        
        dc_prev = dc * f
        
        grads = {
            'W_f': dW_f, 'U_f': dU_f, 'b_f': db_f,
            'W_i': dW_i, 'U_i': dU_i, 'b_i': db_i,
            'W_c': dW_c, 'U_c': dU_c, 'b_c': db_c,
            'W_o': dW_o, 'U_o': dU_o, 'b_o': db_o
        }
        
        return dx, dh_prev, dc_prev, grads
    
    def update_weights(self, grads, learning_rate, reg_lambda):
        """Cập nhật weights"""
        for key, grad in grads.items():
            if hasattr(self, key):
                weight = getattr(self, key)
                weight -= learning_rate * (grad + reg_lambda * weight)
                setattr(self, key, weight)


class LSTMModel:
    """LSTM Model hoàn chỉnh"""
    def __init__(self, input_size, hidden_size, output_size, 
                 learning_rate=0.01, reg_lambda=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        # LSTM cell
        self.cell = LSTMCell(input_size, hidden_size)
        
        # Output layer
        self.W_out = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros((1, output_size))
        
        # Cache cho sequence
        self.h_cache = []
        self.c_cache = []
        self.y_cache = []
        
        print("=== KHỞI TẠO LSTM ===")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Output size: {output_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Regularization: {reg_lambda}")
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X, return_sequences=False):
        """
        X: [batch_size, seq_length, input_size]
        """
        batch_size, seq_length, _ = X.shape
        
        # Khởi tạo hidden và cell state
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        self.h_cache = []
        self.c_cache = []
        self.y_cache = []
        
        # Duyệt qua từng step
        for t in range(seq_length):
            x_t = X[:, t, :]
            h, c = self.cell.forward(x_t, h, c)
            
            # Output
            y_t = np.dot(h, self.W_out) + self.b_out
            
            self.h_cache.append(h)
            self.c_cache.append(c)
            self.y_cache.append(y_t)
        
        if return_sequences:
            return np.stack(self.y_cache, axis=1)
        else:
            # Lấy output cuối cùng và áp dụng softmax
            return self._softmax(self.y_cache[-1])
    
    def backward(self, X, y):
        """Backward pass"""
        batch_size, seq_length, _ = X.shape
        y_one_hot = self.one_hot_encode(y)
        
        # Gradient cho output cuối
        dY = self.y_cache[-1] - y_one_hot  # Cross-entropy + softmax gradient
        dY = dY / batch_size  # Average over batch
        
        # Backward qua output layer
        dW_out = np.dot(self.h_cache[-1].T, dY)
        db_out = np.sum(dY, axis=0, keepdims=True)
        dh_next = np.dot(dY, self.W_out.T)
        dc_next = np.zeros((batch_size, self.hidden_size))
        
        # Tích lũy gradient
        all_grads = {
            'W_f': 0, 'U_f': 0, 'b_f': 0,
            'W_i': 0, 'U_i': 0, 'b_i': 0,
            'W_c': 0, 'U_c': 0, 'b_c': 0,
            'W_o': 0, 'U_o': 0, 'b_o': 0
        }
        
        # Backward qua từng step
        for t in reversed(range(seq_length)):
            dh_next, dc_next, grads = self.cell.backward(dh_next, dc_next)
            
            # Tích lũy gradient
            for key in all_grads:
                all_grads[key] += grads[key]
        
        # Cập nhật weights
        self.W_out -= self.learning_rate * (dW_out + self.reg_lambda * self.W_out)
        self.b_out -= self.learning_rate * db_out
        self.cell.update_weights(all_grads, self.learning_rate, self.reg_lambda)
    
    def one_hot_encode(self, y):
        y_one_hot = np.zeros((y.shape[0], self.output_size))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot
    
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_one_hot = self.one_hot_encode(y_true)
        
        # Cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(y_pred + 1e-15)) / m
        
        # L2 regularization
        reg_loss = 0
        for key in ['W_f', 'U_f', 'W_i', 'U_i', 'W_c', 'U_c', 'W_o', 'U_o']:
            weight = getattr(self.cell, key)
            reg_loss += np.sum(weight ** 2)
        reg_loss += np.sum(self.W_out ** 2)
        
        loss += (self.reg_lambda / (2 * m)) * reg_loss
        return loss
    
    def fit(self, X, y, epochs=100, batch_size=32, verbose=True, patience=20):
        """Huấn luyện model"""
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                y_pred = self.forward(X_batch, return_sequences=False)
                
                # Loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                num_batches += 1
                
                # Backward
                self.backward(X_batch, y_batch)
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                accuracy = self.accuracy(X, y)
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return losses
    
    def predict(self, X):
        y_pred = self.forward(X, return_sequences=False)
        return np.argmax(y_pred, axis=1)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) * 100


# ============================================================
# === MAIN CODE - SỬ DỤNG VỚI DỮ LIỆU CỦA BẠN ===
# ============================================================

if __name__ == "__main__":
    # === 1. LOAD DỮ LIỆU CỦA BẠN ===
    # Bạn thay phần này bằng dữ liệu thực tế của bạn
    # Ví dụ:
    # data = pd.read_csv('your_data.csv')
    # X = data.drop(columns=["tool_type"])
    # y = data["tool_type"]
    
    # === TẠO DỮ LIỆU MẪU (THAY BẰNG DỮ LIỆU THẬT) ===
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Tạo dữ liệu mẫu (bạn thay bằng data thật)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Encode labels (nếu cần)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # === 2. CHUYỂN THÀNH SEQUENCE CHO LSTM ===
    # LSTM cần input dạng [samples, seq_length, features]
    # Với dữ liệu tabular, ta dùng window_size để tạo sequence
    window_size = 5  # Số bước thời gian
    
    def create_sequences(X, y, window_size):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size + 1):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size-1])  # Lấy label của step cuối
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)
    
    print(f"X_train_seq shape: {X_train_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}")
    
    # === 3. KHỞI TẠO VÀ HUẤN LUYỆN LSTM ===
    input_size = X_train_seq.shape[2]
    output_size = len(np.unique(y_train))
    
    lstm_model = LSTMModel(
        input_size=input_size,
        hidden_size=32,
        output_size=output_size,
        learning_rate=0.01,
        reg_lambda=0.001
    )
    
    # Huấn luyện
    lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        verbose=True,
        patience=20
    )
    
    # === 4. ĐÁNH GIÁ ===
    y_pred = lstm_model.predict(X_test_seq)
    acc = accuracy_score(y_test_seq, y_pred)
    print(f"\n=== KẾT QUẢ ===")
    print(f"Test Accuracy: {acc*100:.2f}%")