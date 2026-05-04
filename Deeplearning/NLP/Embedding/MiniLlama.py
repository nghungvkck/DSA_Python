import math
import random
import copy

# -------------------- Helper functions --------------------
def softmax(x):
    """Softmax trên vector 1D (list)"""
    max_val = max(x)
    exp_x = [math.exp(xi - max_val) for xi in x]
    s = sum(exp_x)
    return [ei / s for ei in exp_x]

def matmul_vec_mat(vec, mat):
    """vec (list) dot mat (list of list) -> list"""
    res = [0.0] * len(mat)
    for i in range(len(mat)):
        total = 0.0
        for j in range(len(vec)):
            total += vec[j] * mat[i][j]
        res[i] = total
    return res

def matmul_mat_vec(mat, vec):
    """mat (list of list) * vec (list) -> list (dot each row)"""
    res = [0.0] * len(mat)
    for i in range(len(mat)):
        total = 0.0
        for j in range(len(vec)):
            total += mat[i][j] * vec[j]
        res[i] = total
    return res

def matmul_mat_mat(A, B):
    """A (m x n) * B (n x p) -> m x p (list of list)"""
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            if aik != 0:
                for j in range(p):
                    C[i][j] += aik * B[k][j]
    return C

def elementwise_mul(a, b):
    """a, b là list cùng độ dài -> list"""
    return [a[i] * b[i] for i in range(len(a))]

def add_vectors(a, b):
    return [a[i] + b[i] for i in range(len(a))]

# -------------------- RMSNorm --------------------
class RMSNorm:
    def __init__(self, hidden_size, eps=1e-6):
        self.weight = [1.0] * hidden_size   # gamma, learnable
        self.eps = eps

    def __call__(self, x):
        """x: list of list? (seq_len x hidden_size) -> cùng shape"""
        # x có thể là 2D (seq_len x hidden_size) hoặc 1D (hidden_size)
        if x and isinstance(x[0], list):
            return [self._norm(vec) for vec in x]
        else:
            return self._norm(x)

    def _norm(self, vec):
        variance = sum(v * v for v in vec) / len(vec)
        inv_std = 1.0 / math.sqrt(variance + self.eps)
        out = [self.weight[i] * (vec[i] * inv_std) for i in range(len(vec))]
        return out

# -------------------- Rotary Position Embedding (RoPE) --------------------
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """tạo bảng cos và sin cho RoPE"""
    freqs = [1.0 / (theta ** (2 * (i // 2) / dim)) for i in range(dim)]
    freqs = [[freq] * max_seq_len for freq in freqs]  # dim x max_len
    # tạo ma trận góc: angle(t) = t / (theta^(2i/d))
    angles = [[t * freqs[i][t] for t in range(max_seq_len)] for i in range(dim)]
    cos_cached = [[math.cos(angle) for angle in row] for row in angles]   # dim x max_len
    sin_cached = [[math.sin(angle) for angle in row] for row in angles]
    return cos_cached, sin_cached

def apply_rotary_emb(x, cos, sin):
    """
    x: list of list (seq_len x dim) – thường là Q hoặc K
    cos, sin: list of list (dim x max_seq_len) – nhưng ở đây ta truyền 2D: dim x seq_len
    """
    seq_len = len(x)
    dim = len(x[0])
    out = []
    for t in range(seq_len):
        vec = x[t]
        rotated = [0.0] * dim
        for i in range(0, dim, 2):
            if i+1 >= dim: break
            cos_t = cos[i][t]   # cos cho cặp i
            sin_t = sin[i][t]
            rotated[i]   = vec[i] * cos_t - vec[i+1] * sin_t
            rotated[i+1] = vec[i] * sin_t + vec[i+1] * cos_t
        out.append(rotated)
    return out

# -------------------- Causal Multi‑Head Attention --------------------
class CausalSelfAttention:
    def __init__(self, hidden_size, num_heads, max_seq_len, dropout=0.1):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.Wq = [[random.gauss(0, 0.02) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Wk = [[random.gauss(0, 0.02) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Wv = [[random.gauss(0, 0.02) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Wo = [[random.gauss(0, 0.02) for _ in range(hidden_size)] for _ in range(hidden_size)]

        # RoPE cos/sin bảng (dùng chung cho toàn bộ model)
        self.cos, self.sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        # dropout (bỏ qua vì không dùng numpy)

    def _reshape_heads(self, x, batch=False):
        """x: seq_len x hidden_size -> num_heads x seq_len x head_dim"""
        seq_len = len(x)
        heads = []
        for h in range(self.num_heads):
            head_out = []
            for t in range(seq_len):
                start = h * self.head_dim
                head_vec = x[t][start:start+self.head_dim]
                head_out.append(head_vec)
            heads.append(head_out)
        return heads

    def _combine_heads(self, heads):
        """heads: num_heads x seq_len x head_dim -> seq_len x hidden_size"""
        seq_len = len(heads[0])
        combined = []
        for t in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(heads[h][t])
            combined.append(concat)
        return combined

    def forward(self, x, mask=None):
        """
        x: list of list (seq_len x hidden_size)
        mask: có thể là causal mask (list of list) – ở đây tự tính
        """
        seq_len = len(x)
        # Linear projections
        q = matmul_mat_vec(self.Wq, x)   # seq_len x hidden_size
        k = matmul_mat_vec(self.Wk, x)
        v = matmul_mat_vec(self.Wv, x)

        # Reshape thành heads
        q_heads = self._reshape_heads(q)   # num_heads x seq_len x head_dim
        k_heads = self._reshape_heads(k)
        v_heads = self._reshape_heads(v)

        # Apply RoPE trên các head (riêng lẻ)
        for h in range(self.num_heads):
            q_heads[h] = apply_rotary_emb(q_heads[h], self.cos, self.sin)
            k_heads[h] = apply_rotary_emb(k_heads[h], self.cos, self.sin)

        # Attention
        attn_outs = []
        for h in range(self.num_heads):
            # Tính scores: Q * K^T / sqrt(d_k)
            scores = [[0.0]*seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    dot = 0.0
                    for d in range(self.head_dim):
                        dot += q_heads[h][i][d] * k_heads[h][j][d]
                    scores[i][j] = dot / math.sqrt(self.head_dim)
            # Causal mask: cấm nhìn về tương lai
            for i in range(seq_len):
                for j in range(i+1, seq_len):
                    scores[i][j] = -1e9
            # Softmax
            attn_weights = [softmax(row) for row in scores]
            # Tính O = attn * V
            out_h = []
            for i in range(seq_len):
                out_vec = [0.0]*self.head_dim
                for j in range(seq_len):
                    w = attn_weights[i][j]
                    vj = v_heads[h][j]
                    for d in range(self.head_dim):
                        out_vec[d] += w * vj[d]
                out_h.append(out_vec)
            attn_outs.append(out_h)

        # Ghép các head
        combined = self._combine_heads(attn_outs)   # seq_len x hidden_size
        # Projection cuối
        output = matmul_mat_vec(self.Wo, combined)
        return output

# -------------------- Feed‑Forward với SwiGLU --------------------
class SwiGLUFeedForward:
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        self.W_gate = [[random.gauss(0, 0.02) for _ in range(hidden_size)] for _ in range(intermediate_size)]
        self.W_up   = [[random.gauss(0, 0.02) for _ in range(hidden_size)] for _ in range(intermediate_size)]
        self.W_down = [[random.gauss(0, 0.02) for _ in range(intermediate_size)] for _ in range(hidden_size)]

    def _swish(self, x):
        return x * (1.0 / (1.0 + math.exp(-x)))   # sigmoid

    def forward(self, x):
        # x: list of vectors [seq_len x hidden_size]
        out = []
        for vec in x:
            gate = matmul_mat_vec(self.W_gate, vec)  # intermediate_size
            up   = matmul_mat_vec(self.W_up, vec)
            gate_swish = [self._swish(g) for g in gate]
            # elementwise mul
            hidden = [gate_swish[i] * up[i] for i in range(len(gate))]
            down = matmul_mat_vec(self.W_down, hidden)  # hidden_size
            out.append(down)
        return out

# -------------------- Transformer Block (Decoder) --------------------
class LlamaBlock:
    def __init__(self, hidden_size, num_heads, max_seq_len, intermediate_size, dropout=0.1):
        self.attn = CausalSelfAttention(hidden_size, num_heads, max_seq_len, dropout)
        self.ff = SwiGLUFeedForward(hidden_size, intermediate_size, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, x):
        # Pre‑norm + self‑attention
        normed = self.norm1(x)
        attn_out = self.attn.forward(normed)
        x = add_vectors(x, attn_out) if isinstance(x[0], list) else [add_vectors(x[i], attn_out[i]) for i in range(len(x))]
        # Pre‑norm + FF
        normed2 = self.norm2(x)
        ff_out = self.ff.forward(normed2)
        x = add_vectors(x, ff_out) if isinstance(x[0], list) else [add_vectors(x[i], ff_out[i]) for i in range(len(x))]
        return x

# -------------------- MiniLlama Model --------------------
class MiniLlama:
    def __init__(self, vocab_size=100, hidden_size=64, num_heads=4, num_layers=3,
                 max_seq_len=32, intermediate_size=None):
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Token embedding (không dùng bias, theo Llama)
        self.token_embedding = [[random.gauss(0, 0.02) for _ in range(hidden_size)] for _ in range(vocab_size)]
        self.blocks = [LlamaBlock(hidden_size, num_heads, max_seq_len, intermediate_size) for _ in range(num_layers)]
        self.norm_final = RMSNorm(hidden_size)
        self.lm_head = [[random.gauss(0, 0.02) for _ in range(hidden_size)] for _ in range(vocab_size)]   # không bias

    def forward(self, input_ids):
        """
        input_ids: list of int (seq_len) – mô hình tự hồi quy, batch=1 cho đơn giản
        Trả về logits (seq_len x vocab_size)
        """
        seq_len = len(input_ids)
        # Token embedding
        h = [self.token_embedding[token_id][:] for token_id in input_ids]   # seq_len x hidden_size

        # Chạy qua các block
        for block in self.blocks:
            h = block.forward(h)

        # Final RMSNorm
        h = self.norm_final(h)   # seq_len x hidden_size

        # LM head: linear không bias
        logits = []
        for vec in h:
            logit_vec = [0.0]*self.vocab_size
            for i in range(self.vocab_size):
                # dot product
                total = 0.0
                for j in range(self.hidden_size):
                    total += self.lm_head[i][j] * vec[j]
                logit_vec[i] = total
            logits.append(logit_vec)
        return logits

    def generate(self, start_ids, max_new_tokens, temperature=1.0):
        """
        Sinh văn bản tự hồi quy.
        start_ids: list các token id ban đầu
        """
        ids = start_ids[:]
        for _ in range(max_new_tokens):
            logits = self.forward(ids)           # (seq_len x vocab_size)
            logits_last = logits[-1]             # chỉ lấy token cuối
            # Áp dụng temperature
            if temperature > 0:
                scaled = [l / temperature for l in logits_last]
                probs = softmax(scaled)
                # random sample
                r = random.random()
                cum = 0.0
                next_id = None
                for token_id, p in enumerate(probs):
                    cum += p
                    if r < cum:
                        next_id = token_id
                        break
                if next_id is None:
                    next_id = len(probs)-1
            else:
                # greedy
                next_id = max(range(self.vocab_size), key=lambda i: logits_last[i])
            ids.append(next_id)
        return ids


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Tạo mô hình MiniLlama siêu nhỏ
    llama = MiniLlama(vocab_size=50, hidden_size=32, num_heads=2, num_layers=2, max_seq_len=20)

    # Mã hoá câu example (giả sử token ids)
    prompt = [5, 12, 8, 2]   # "hello world"
    print("Input ids:", prompt)

    # Forward: lấy logits
    logits = llama.forward(prompt)
    print("\nLogits shape: (seq_len={}, vocab_size={})".format(len(logits), len(logits[0])))
    print("Logits cho token cuối (5 chiều đầu):", logits[-1][:5])

    # Sinh thêm 3 token
    generated = llama.generate(prompt, max_new_tokens=3, temperature=0.8)
    print("\nGenerated ids:", generated)