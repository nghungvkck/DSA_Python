import math
import random

# ==================== CÁC HÀM TIỆN ÍCH ====================
def softmax(x):
    """Softmax 1D"""
    m = max(x)
    e = [math.exp(xi - m) for xi in x]
    s = sum(e)
    return [ei / s for ei in e]

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer norm trên vector 1D x"""
    mean = sum(x) / len(x)
    var = sum((xi - mean) ** 2 for xi in x) / len(x)
    inv_std = 1.0 / math.sqrt(var + eps)
    return [gamma[i] * (x[i] - mean) * inv_std + beta[i] for i in range(len(x))]

def linear(x, W, b):
    """y = W^T x + b; x là vector 1D"""
    return [b[j] + sum(x[i] * W[i][j] for i in range(len(x))) for j in range(len(b))]

def add_residual(x, residual):
    """Cộng hai vector cùng độ dài"""
    return [x[i] + residual[i] for i in range(len(x))]

# ==================== POSITIONAL ENCODING ====================
def get_positional_encoding(seq_len, d_model):
    """Ma trận PE (seq_len x d_model)"""
    pe = [[0.0] * d_model for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos][i] = math.sin(pos / (10000 ** (2 * i / d_model)))
            if i + 1 < d_model:
                pe[pos][i+1] = math.cos(pos / (10000 ** (2 * i / d_model)))
    return pe

# ==================== MULTI‑HEAD ATTENTION ====================
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Khởi tạo trọng số
        self.Wq = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.Wk = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.Wv = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.Wo = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.bias = [0.0] * d_model

    def _split_heads(self, x):
        """x: seq_len x d_model -> num_heads x seq_len x head_dim"""
        seq_len = len(x)
        heads = []
        for h in range(self.num_heads):
            head = []
            for t in range(seq_len):
                start = h * self.head_dim
                head.append(x[t][start:start + self.head_dim])
            heads.append(head)
        return heads

    def _combine_heads(self, heads):
        """heads: num_heads x seq_len x head_dim -> seq_len x d_model"""
        seq_len = len(heads[0])
        combined = []
        for t in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][t])
            combined.append(concat)
        return combined

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: list of list (seq_len x d_model)
        mask: ma trận (seq_len_q x seq_len_k) với 0 ở những vị trí được phép, -inf ở vị trí bị cấm (cho decoder)
        """
        seq_len = len(query)
        # Linear projections
        Q = linear(query, self.Wq, self.bias)   # seq_len x d_model
        K = linear(key, self.Wk, self.bias)
        V = linear(value, self.Wv, self.bias)

        # Split heads
        Q_heads = self._split_heads(Q)   # [num_heads x seq_len x head_dim]
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        # Tính attention cho từng head
        attn_outputs = []
        for h in range(self.num_heads):
            # scores: seq_len x seq_len
            scores = [[0.0] * seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    dot = sum(Q_heads[h][i][d] * K_heads[h][j][d] for d in range(self.head_dim))
                    scores[i][j] = dot / math.sqrt(self.head_dim)

            if mask is not None:
                for i in range(seq_len):
                    for j in range(seq_len):
                        scores[i][j] += mask[i][j]   # mask chứa -inf ở vị trí cần che

            # Softmax theo từng hàng
            attn_weights = [softmax(row) for row in scores]

            # Nhân với V
            out_h = []
            for i in range(seq_len):
                vec = [0.0] * self.head_dim
                for j in range(seq_len):
                    w = attn_weights[i][j]
                    vj = V_heads[h][j]
                    for d in range(self.head_dim):
                        vec[d] += w * vj[d]
                out_h.append(vec)
            attn_outputs.append(out_h)

        # Ghép các head
        combined = self._combine_heads(attn_outputs)   # seq_len x d_model
        output = linear(combined, self.Wo, self.bias)
        return output

# ==================== FEED‑FORWARD NETWORK ====================
class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_ff)]
        self.b1 = [0.0] * d_ff
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(d_ff)] for _ in range(d_model)]
        self.b2 = [0.0] * d_model

    def relu(self, x):
        return max(0, x)

    def forward(self, x):
        """x: list of vector (seq_len x d_model)"""
        out = []
        for vec in x:
            hidden = linear(vec, self.W1, self.b1)          # d_ff
            hidden = [self.relu(v) for v in hidden]
            out_vec = linear(hidden, self.W2, self.b2)      # d_model
            out.append(out_vec)
        return out

# ==================== ENCODER LAYER ====================
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1_gamma = [1.0] * d_model
        self.norm1_beta = [0.0] * d_model
        self.norm2_gamma = [1.0] * d_model
        self.norm2_beta = [0.0] * d_model

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out = self.self_attn.forward(x, x, x, mask)
        # Residual
        x = [add_residual(attn_out[i], x[i]) for i in range(len(x))]
        # Layer norm
        x = [layer_norm(vec, self.norm1_gamma, self.norm1_beta) for vec in x]
        # Feed-forward + residual + norm
        ff_out = self.ff.forward(x)
        x = [add_residual(ff_out[i], x[i]) for i in range(len(x))]
        x = [layer_norm(vec, self.norm2_gamma, self.norm2_beta) for vec in x]
        return x

# ==================== DECODER LAYER ====================
class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1_gamma = [1.0] * d_model
        self.norm1_beta = [0.0] * d_model
        self.norm2_gamma = [1.0] * d_model
        self.norm2_beta = [0.0] * d_model
        self.norm3_gamma = [1.0] * d_model
        self.norm3_beta = [0.0] * d_model

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        # Masked self-attention
        attn_out = self.self_attn.forward(x, x, x, self_mask)
        x = [add_residual(attn_out[i], x[i]) for i in range(len(x))]
        x = [layer_norm(vec, self.norm1_gamma, self.norm1_beta) for vec in x]

        # Cross-attention (query từ decoder, key/value từ encoder)
        cross_out = self.cross_attn.forward(x, enc_output, enc_output, cross_mask)
        x = [add_residual(cross_out[i], x[i]) for i in range(len(x))]
        x = [layer_norm(vec, self.norm2_gamma, self.norm2_beta) for vec in x]

        # Feed-forward
        ff_out = self.ff.forward(x)
        x = [add_residual(ff_out[i], x[i]) for i in range(len(x))]
        x = [layer_norm(vec, self.norm3_gamma, self.norm3_beta) for vec in x]
        return x

# ==================== TRANSFORMER ====================
class Transformer:
    def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, num_decoder_layers,
                 max_seq_len, vocab_size_tgt):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.pe = get_positional_encoding(max_seq_len, d_model)

        # Encoder layers
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_encoder_layers)]

        # Decoder layers
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_decoder_layers)]

        # Final linear layer (để ánh xạ sang từ vựng)
        self.out_linear = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(vocab_size_tgt)]
        self.out_bias = [0.0] * vocab_size_tgt

    def _add_pe(self, emb_seq):
        """Cộng positional encoding vào embedding sequence"""
        seq_len = len(emb_seq)
        return [[emb_seq[i][j] + self.pe[i][j] for j in range(self.d_model)] for i in range(seq_len)]

    def _causal_mask(self, size):
        """Tạo causal mask (tam giác trên) cho decoder self-attention"""
        mask = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i+1, size):
                mask[i][j] = -1e9
        return mask

    def encode(self, src_emb):
        """
        src_emb: list of embedding vectors (src_seq_len x d_model) cho câu nguồn
        Trả về output của encoder (cùng shape)
        """
        # Thêm positional encoding
        x = self._add_pe(src_emb)
        for layer in self.encoder_layers:
            x = layer.forward(x)
        return x

    def decode(self, tgt_emb, enc_output):
        """
        tgt_emb: list of embedding vectors (tgt_seq_len x d_model) cho câu đích (đã shift phải)
        enc_output: output từ encoder
        Trả về logits (tgt_seq_len x vocab_size_tgt)
        """
        seq_len = len(tgt_emb)
        # Thêm positional encoding
        x = self._add_pe(tgt_emb)
        # Causal mask cho self-attention
        causal_mask = self._causal_mask(seq_len)
        for layer in self.decoder_layers:
            x = layer.forward(x, enc_output, self_mask=causal_mask, cross_mask=None)
        # Linear ra vocab
        logits = []
        for vec in x:
            logit_vec = linear(vec, self.out_linear, self.out_bias)
            logits.append(logit_vec)
        return logits

    def forward(self, src_emb, tgt_emb):
        """Forward toàn bộ: encode rồi decode"""
        enc_out = self.encode(src_emb)
        logits = self.decode(tgt_emb, enc_out)
        return logits

# ==================== VÍ DỤ MINH HỌA ====================
if __name__ == "__main__":
    # Cấu hình
    d_model = 8          # số chiều embedding
    num_heads = 2
    d_ff = 16
    num_encoder_layers = 2
    num_decoder_layers = 2
    max_seq_len = 10
    vocab_size_tgt = 20   # kích thước từ vựng đích (cho output)

    # Dữ liệu giả: embedding cho câu nguồn (3 token) và câu đích (4 token)
    src_len = 3
    tgt_len = 4
    src_emb = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(src_len)]
    tgt_emb = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(tgt_len)]

    print("Source embedding sequence (3x8):")
    for i, vec in enumerate(src_emb):
        print(f"Token {i+1}: {[round(x,2) for x in vec]}")

    # Khởi tạo Transformer
    model = Transformer(d_model, num_heads, d_ff, num_encoder_layers, num_decoder_layers, max_seq_len, vocab_size_tgt)

    # Forward
    logits = model.forward(src_emb, tgt_emb)
    print("\nLogits output (tgt_seq_len=4, vocab_size=20):")
    for i, logit_vec in enumerate(logits):
        print(f"Token {i+1} logits (5 chiều đầu): {[round(x,2) for x in logit_vec[:5]]}")

    # Áp dụng softmax để ra xác suất (nếu cần)
    probs = [softmax(vec) for vec in logits]
    print("\nXác suất token đầu tiên (10 chiều):", [round(p,3) for p in probs[0][:10]], "...")