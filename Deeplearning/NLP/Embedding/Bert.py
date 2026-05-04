import math
import random

class MiniBERT:
    """
    Mô hình BERT cực nhỏ (miniBERT) với các thành phần chính:
    - Token embedding + Positional embedding + Segment embedding
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Stack các Transformer encoder block
    - Masked Language Model head (đầu ra cho từng token)
    """
    def __init__(self, vocab_size=1000, max_len=128, hidden_size=128, num_heads=4,
                 num_layers=3, intermediate_size=256, dropout=0.1):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout

        # Khởi tạo các tham số
        self.token_embedding = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(vocab_size)]
        self.position_embedding = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(max_len)]
        self.segment_embedding = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(2)]

        # Các layer encoder
        self.encoders = []
        for _ in range(num_layers):
            attn = MultiHeadAttention(hidden_size, num_heads, dropout)
            ff = FeedForward(hidden_size, intermediate_size, dropout)
            norm1 = LayerNorm(hidden_size)
            norm2 = LayerNorm(hidden_size)
            self.encoders.append((attn, ff, norm1, norm2))

        # LM head: từ hidden_size -> vocab_size
        self.lm_head = Linear(hidden_size, vocab_size)
        # LayerNorm cuối
        self.lm_norm = LayerNorm(hidden_size)

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        input_ids: list of list (batch_size x seq_len)
        segment_ids: list of list (0/1), mặc định toàn 0
        attention_mask: list of list (0/1), 1 là token được attention
        Trả về logits cho MLM: (batch_size x seq_len x vocab_size)
        """
        batch_size = len(input_ids)
        seq_len = len(input_ids[0])

        if segment_ids is None:
            segment_ids = [[0] * seq_len for _ in range(batch_size)]
        if attention_mask is None:
            attention_mask = [[1] * seq_len for _ in range(batch_size)]

        # 1. Embedding sum
        hidden_states = []
        for b in range(batch_size):
            seq_emb = []
            for t in range(seq_len):
                token_id = input_ids[b][t]
                seg_id = segment_ids[b][t]
                # token emb
                token_vec = self.token_embedding[token_id][:]
                # position emb
                pos_vec = self.position_embedding[t][:]
                # segment emb
                seg_vec = self.segment_embedding[seg_id][:]
                # cộng dồn
                emb = [token_vec[i] + pos_vec[i] + seg_vec[i] for i in range(self.hidden_size)]
                seq_emb.append(emb)
            hidden_states.append(seq_emb)

        # 2. Dropout (giả lập: bỏ qua hoặc nhân với 1/(1-p) ngẫu nhiên)
        # Thực tế bỏ qua để đơn giản

        # 3. Chạy qua các encoder layer
        for attn, ff, norm1, norm2 in self.encoders:
            new_hidden = []
            for b in range(batch_size):
                # Attention + residual
                attn_out = attn.forward(hidden_states[b], hidden_states[b], hidden_states[b], attention_mask[b])
                # Add & norm
                attn_out = norm1.forward(attn_out, hidden_states[b])  # hidden_states[b] là residual
                # Feed-forward
                ff_out = ff.forward(attn_out)
                # Add & norm
                ff_out = norm2.forward(ff_out, attn_out)
                new_hidden.append(ff_out)
            hidden_states = new_hidden

        # 4. LM head
        logits = []
        for b in range(batch_size):
            seq_logits = []
            for t in range(seq_len):
                # LayerNorm trước head
                norm_vec = self.lm_norm.forward(hidden_states[b][t])
                # Linear -> logits
                logit_vec = self.lm_head.forward(norm_vec)
                seq_logits.append(logit_vec)
            logits.append(seq_logits)
        return logits  # list [batch][seq][vocab_size]


# ---------- Các thành phần cơ bản ----------
class MultiHeadAttention:
    def __init__(self, hidden_size, num_heads, dropout):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.W_q = Linear(hidden_size, hidden_size)
        self.W_k = Linear(hidden_size, hidden_size)
        self.W_v = Linear(hidden_size, hidden_size)
        self.W_o = Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        # query, key, value: list of list [seq_len x hidden_size]
        seq_len = len(query)
        batch_mode = isinstance(query[0], list)  # nếu đã có batch dimension
        # Đơn giản hóa: làm trên 1 sequence
        Q = self.W_q.forward(query)   # seq_len x hidden_size
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)

        # Split thành nhiều head: (seq_len, num_heads, head_dim)
        def split_heads(x):
            result = []
            for i in range(seq_len):
                heads = []
                for h in range(self.num_heads):
                    start = h * self.head_dim
                    head_vec = x[i][start:start+self.head_dim]
                    heads.append(head_vec)
                result.append(heads)
            # chuyển thành (num_heads, seq_len, head_dim) cho dễ tính
            out = [[[0]*self.head_dim for _ in range(seq_len)] for _ in range(self.num_heads)]
            for h in range(self.num_heads):
                for s in range(seq_len):
                    out[h][s] = result[s][h]
            return out
        Q_heads = split_heads(Q)
        K_heads = split_heads(K)
        V_heads = split_heads(V)

        # Attention score cho từng head
        attn_outputs = []
        for h in range(self.num_heads):
            # Tính ma trận attention: Q * K^T / sqrt(d_k)
            scores = [[0.0]*seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    dot = 0.0
                    for d in range(self.head_dim):
                        dot += Q_heads[h][i][d] * K_heads[h][j][d]
                    scores[i][j] = dot / math.sqrt(self.head_dim)
            # Áp dụng mask (1 là được attend, 0 là không)
            if mask is not None:
                for i in range(seq_len):
                    for j in range(seq_len):
                        if mask[j] == 0:  # mask trên key? Trong BERT, mask trên key vị trí padding
                            scores[i][j] = -1e9
            # Softmax theo từng hàng
            attn_weights = []
            for i in range(seq_len):
                row = scores[i]
                # softmax
                max_val = max(row)
                exp_row = [math.exp(x - max_val) for x in row]
                sum_exp = sum(exp_row)
                soft_row = [e / sum_exp for e in exp_row]
                attn_weights.append(soft_row)
            # Tính output cho head: attn_weights * V
            out_heads = []
            for i in range(seq_len):
                vec = [0.0]*self.head_dim
                for j in range(seq_len):
                    w = attn_weights[i][j]
                    vj = V_heads[h][j]
                    for d in range(self.head_dim):
                        vec[d] += w * vj[d]
                out_heads.append(vec)
            attn_outputs.append(out_heads)

        # Ghép các head: (seq_len, num_heads*head_dim) = (seq_len, hidden_size)
        combined = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(attn_outputs[h][i])
            combined.append(concat)

        # Projection cuối
        output = self.W_o.forward(combined)
        return output

class FeedForward:
    def __init__(self, hidden_size, intermediate_size, dropout):
        self.linear1 = Linear(hidden_size, intermediate_size)
        self.linear2 = Linear(intermediate_size, hidden_size)
        self.dropout = dropout

    def forward(self, x):
        # x: list of list [seq_len x hidden_size]
        intermediate = self.linear1.forward(x)
        # GELU activation (xấp xỉ)
        gelu_out = []
        for i in range(len(intermediate)):
            row = [self._gelu(val) for val in intermediate[i]]
            gelu_out.append(row)
        output = self.linear2.forward(gelu_out)
        return output

    def _gelu(self, x):
        # xấp xỉ tanh
        return 0.5 * x * (1 + math.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

class LayerNorm:
    def __init__(self, hidden_size, eps=1e-5):
        self.gamma = [1.0] * hidden_size
        self.beta = [0.0] * hidden_size
        self.eps = eps

    def forward(self, x, residual=None):
        # x: list of list [batch? seq_len x hidden_size] hoặc [seq_len x hidden_size]
        # nếu có residual thì thực hiện add trước (dùng cho pre-norm)
        if residual is not None:
            # cộng residual
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j] += residual[i][j]
        # LayerNorm chuẩn
        for i in range(len(x)):
            mean = sum(x[i]) / len(x[i])
            var = sum((val - mean)**2 for val in x[i]) / len(x[i])
            inv_std = 1.0 / math.sqrt(var + self.eps)
            for j in range(len(x[i])):
                x[i][j] = self.gamma[j] * (x[i][j] - mean) * inv_std + self.beta[j]
        return x

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = [[random.uniform(-0.1, 0.1) for _ in range(in_features)] for _ in range(out_features)]
        self.bias = [0.0] * out_features

    def forward(self, x):
        # x: list of list [batch? seq_len x in_features] hoặc [in_features]
        if isinstance(x[0], list):  # nhiều vector
            out = []
            for vec in x:
                out_vec = []
                for j in range(len(self.weight)):
                    s = self.bias[j]
                    for k in range(len(vec)):
                        s += self.weight[j][k] * vec[k]
                    out_vec.append(s)
                out.append(out_vec)
            return out
        else:
            out_vec = []
            for j in range(len(self.weight)):
                s = self.bias[j]
                for k in range(len(x)):
                    s += self.weight[j][k] * x[k]
                out_vec.append(s)
            return out_vec


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Tạo mô hình miniBERT
    bert = MiniBERT(vocab_size=100, max_len=20, hidden_size=16, num_heads=2, num_layers=2, intermediate_size=32)

    # Dữ liệu giả: batch=2, seq_len=4
    input_ids = [
        [5, 12, 8, 2],   # câu 1
        [1, 3, 9, 7]     # câu 2
    ]
    segment_ids = [[0,0,0,0], [0,0,0,0]]
    attention_mask = [[1,1,1,1], [1,1,1,0]]   # token cuối câu 2 bị mask

    # Forward pass
    logits = bert.forward(input_ids, segment_ids, attention_mask)

    print("Kích thước logits:", len(logits), "x", len(logits[0]), "x", len(logits[0][0]))
    print("Logits cho câu 1, token đầu tiên (5 chiều đầu):", logits[0][0][:5])