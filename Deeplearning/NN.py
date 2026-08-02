import numpy as np 

class NeuraNetWorkImproved:
    def __init__ ( self, layers, learning_rate=0.1, reg_lambda = 0.001):

        # layers là số neuron mỗi lớp
        self.layers = layers
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.weights = []
        self.biases = []

        # tạo ma trận
        for i in range(len(layers) -1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        print("=== KHỞI TẠO MẠNG ===")
        print(f"Kiến trúc: {layers}")
        print(f"Weights shapes: {[w.shape for w in self.weights]}")
        print(f"Biases shapes: {[b.shape for b in self.biases]}")
        print(f"Learning rate: {learning_rate}")
        print(f"Regularization: {reg_lambda}")


    # Hàm kích hoạt
    def activate(self, z, activation_type='relu'):
        if activation_type == 'relu':
            return np.maximum(0, z)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation_type == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis = 1, keepdims=True)
        elif activation_type == 'tanh':
            return np.tanh(z)

    # đạo hàm
    def activation_derivative(self, z, activation_type='relu'):
        if activation_type == 'relu':
            return ( z > 0).astype(float)
        elif activation_type == 'sigmoid':
            return z * (1 - z)
        elif activation_type == 'tanh':
            return 1 - z ** 2

    def forward(self, X):
        self.activations = [X]  # chứa output của từng layer
        self.z_values = [] # chứa giá trị trước activation
        current_input = X  # bắt đầu với input 

        #  hidden layer
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_input = self.activate(z, 'relu')
            self.activations.append(current_input)

        z = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.activate(z, 'softmax') # Changed sigmoid to softmax for multi-class classification
        self.activations.append(output)
        return output

    def backward(self, X, y):
        m = X.shape[0]
        y_one_hot  = self.one_hot_encode(y)

        dA = self.activations[-1] - y_one_hot
        dZ = dA
        # Corrected L2 regularization derivative: (reg_lambda / m) * w
        dW = np.dot(self.activations[-2].T, dZ) / m + (self.reg_lambda * self.weights[-1]) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients = [(dW, db)]

        for i in range(len(self.weights) - 1, 0, -1):
            dA_prev = np.dot(dZ, self.weights[i].T)
            dZ = dA_prev * self.activation_derivative(self.z_values[i-1], 'relu')
            # Corrected L2 regularization derivative
            dW = np.dot(self.activations[i-1].T, dZ) / m + (self.reg_lambda * self.weights[i-1]) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            gradients.append((dW, db))

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients[len(self.weights)-1-i][0]
            self.biases[i] -= self.learning_rate * gradients[len(self.weights)-1-i][1]

    def one_hot_encode(self, y):
        n_classes = self.layers[-1]
        y_one_hot = np.zeros((y.shape[0], n_classes))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_one_hot = self.one_hot_encode(y_true)
        loss = -np.sum(y_one_hot * np.log(y_pred + 1e-15)) / m

        reg_loss = 0
        for w in self.weights:
            reg_loss += np.sum(w**2)
        loss += (self.reg_lambda / (2 * m)) * reg_loss
        return loss

    def fit(self, X, y, epochs=1000, verbose=True, patience=50):
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            y_pred = self.forward(X)                        # Forward
            loss = self.compute_loss(y_pred, y)             # Loss
            losses.append(loss)
            self.backward(X, y)                             # Backward
            
            if loss < best_loss:                            # Early stopping
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                accuracy = self.accuracy(X, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return losses

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) * 100

# === MAIN CODE ===
if __name__ == "__main__": 
    X_train  = None
    X_test = None
    y_train  = None
    y_test  = None

    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    
    # === THỬ NHIỀU CẤU HÌNH ===
    configs = [
        {"layers": [input_size, 16, 8, output_size], "lr": 0.1, "epochs": 500},
        {"layers": [input_size, 32, 16, 8, output_size], "lr": 0.05, "epochs": 500},
        {"layers": [input_size, 12, 6, output_size], "lr": 0.1, "epochs": 500},
    ]
    
    best_acc = 0
    best_nn = None
    
    for i, config in enumerate(configs):
        print(f"\n=== CẤU HÌNH {i+1}: layers={config['layers']}, lr={config['lr']} ===")
        
        nn = NeuraNetWorkImproved(
            layers=config['layers'],
            learning_rate=config['lr'],
            reg_lambda=0.001
        )
        
        nn.fit(X_train, y_train, epochs=config['epochs'], verbose=True)
        
        # Evaluate
        y_pred = nn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_nn = nn
    
    print(f"\n=== KẾT QUẢ TỐT NHẤT ===")
    print(f"Accuracy: {best_acc*100:.2f}%")