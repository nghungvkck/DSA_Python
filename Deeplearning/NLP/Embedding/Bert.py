import math
import random


class MiniBERT:
    """
    Mo hinh BERT cuc nho (miniBERT) voi cac thanh phan chinh:
    - Token embedding + positional embedding + segment embedding
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Stack cac Transformer encoder block
    - Masked Language Model head
    """

    def __init__(
        self,
        vocab_size=1000,
        max_len=128,
        hidden_size=128,
        num_heads=4,
        num_layers=3,
        intermediate_size=256,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout

        self.token_embedding = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(vocab_size)
        ]
        self.position_embedding = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(max_len)
        ]
        self.segment_embedding = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(2)
        ]

        self.encoders = []
        for _ in range(num_layers):
            attn = MultiHeadAttention(hidden_size, num_heads, dropout)
            ff = FeedForward(hidden_size, intermediate_size, dropout)
            norm1 = LayerNorm(hidden_size)
            norm2 = LayerNorm(hidden_size)
            self.encoders.append((attn, ff, norm1, norm2))

        self.lm_head = Linear(hidden_size, vocab_size)
        self.lm_norm = LayerNorm(hidden_size)

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        input_ids: list[list[int]] with shape (batch_size, seq_len)
        segment_ids: list[list[int]] with values 0/1
        attention_mask: list[list[int]] with values 0/1
        return: list[batch][seq][vocab_size]
        """
        if not input_ids or not input_ids[0]:
            raise ValueError("input_ids must be a non-empty batch of non-empty sequences")

        batch_size = len(input_ids)
        seq_len = len(input_ids[0])

        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")
        if any(len(seq) != seq_len for seq in input_ids):
            raise ValueError("All sequences in input_ids must have the same length")

        if segment_ids is None:
            segment_ids = [[0] * seq_len for _ in range(batch_size)]
        if attention_mask is None:
            attention_mask = [[1] * seq_len for _ in range(batch_size)]

        if len(segment_ids) != batch_size or any(len(seq) != seq_len for seq in segment_ids):
            raise ValueError("segment_ids must have the same shape as input_ids")
        if len(attention_mask) != batch_size or any(len(seq) != seq_len for seq in attention_mask):
            raise ValueError("attention_mask must have the same shape as input_ids")

        hidden_states = []
        for b in range(batch_size):
            seq_emb = []
            for t in range(seq_len):
                token_id = input_ids[b][t]
                seg_id = segment_ids[b][t]

                if not 0 <= token_id < self.vocab_size:
                    raise ValueError(
                        f"token_id {token_id} is out of range for vocab_size={self.vocab_size}"
                    )
                if seg_id not in (0, 1):
                    raise ValueError("segment_ids values must be 0 or 1")

                token_vec = self.token_embedding[token_id][:]
                pos_vec = self.position_embedding[t][:]
                seg_vec = self.segment_embedding[seg_id][:]
                emb = [
                    token_vec[i] + pos_vec[i] + seg_vec[i]
                    for i in range(self.hidden_size)
                ]
                seq_emb.append(emb)
            hidden_states.append(seq_emb)

        for attn, ff, norm1, norm2 in self.encoders:
            new_hidden = []
            for b in range(batch_size):
                attn_out = attn.forward(
                    hidden_states[b],
                    hidden_states[b],
                    hidden_states[b],
                    attention_mask[b],
                )
                attn_out = norm1.forward(attn_out, hidden_states[b])
                ff_out = ff.forward(attn_out)
                ff_out = norm2.forward(ff_out, attn_out)
                new_hidden.append(ff_out)
            hidden_states = new_hidden

        logits = []
        for b in range(batch_size):
            seq_logits = []
            for t in range(seq_len):
                norm_vec = self.lm_norm.forward(hidden_states[b][t])
                logit_vec = self.lm_head.forward(norm_vec)
                seq_logits.append(logit_vec)
            logits.append(seq_logits)
        return logits


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
        seq_len = len(query)
        Q = self.W_q.forward(query)
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)

        def split_heads(x):
            result = []
            for i in range(seq_len):
                heads = []
                for h in range(self.num_heads):
                    start = h * self.head_dim
                    heads.append(x[i][start : start + self.head_dim])
                result.append(heads)

            out = [
                [[0.0] * self.head_dim for _ in range(seq_len)]
                for _ in range(self.num_heads)
            ]
            for h in range(self.num_heads):
                for s in range(seq_len):
                    out[h][s] = result[s][h]
            return out

        Q_heads = split_heads(Q)
        K_heads = split_heads(K)
        V_heads = split_heads(V)

        attn_outputs = []
        for h in range(self.num_heads):
            scores = [[0.0] * seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    dot = 0.0
                    for d in range(self.head_dim):
                        dot += Q_heads[h][i][d] * K_heads[h][j][d]
                    scores[i][j] = dot / math.sqrt(self.head_dim)

            if mask is not None:
                for i in range(seq_len):
                    for j in range(seq_len):
                        if mask[j] == 0:
                            scores[i][j] = -1e9

            attn_weights = []
            for i in range(seq_len):
                row = scores[i]
                max_val = max(row)
                exp_row = [math.exp(x - max_val) for x in row]
                sum_exp = sum(exp_row)
                attn_weights.append([e / sum_exp for e in exp_row])

            out_heads = []
            for i in range(seq_len):
                vec = [0.0] * self.head_dim
                for j in range(seq_len):
                    weight = attn_weights[i][j]
                    value_vec = V_heads[h][j]
                    for d in range(self.head_dim):
                        vec[d] += weight * value_vec[d]
                out_heads.append(vec)
            attn_outputs.append(out_heads)

        combined = []
        for i in range(seq_len):
            concat = []
            for h in range(self.num_heads):
                concat.extend(attn_outputs[h][i])
            combined.append(concat)

        return self.W_o.forward(combined)


class FeedForward:
    def __init__(self, hidden_size, intermediate_size, dropout):
        self.linear1 = Linear(hidden_size, intermediate_size)
        self.linear2 = Linear(intermediate_size, hidden_size)
        self.dropout = dropout

    def forward(self, x):
        intermediate = self.linear1.forward(x)
        gelu_out = []
        for row in intermediate:
            gelu_out.append([self._gelu(val) for val in row])
        return self.linear2.forward(gelu_out)

    def _gelu(self, x):
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    def __init__(self, hidden_size, eps=1e-5):
        self.gamma = [1.0] * hidden_size
        self.beta = [0.0] * hidden_size
        self.eps = eps

    def forward(self, x, residual=None):
        is_vector = not isinstance(x[0], list)
        if is_vector:
            x = [x[:]]
            if residual is not None:
                residual = [residual[:]]

        if residual is not None:
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j] += residual[i][j]

        for i in range(len(x)):
            mean = sum(x[i]) / len(x[i])
            var = sum((val - mean) ** 2 for val in x[i]) / len(x[i])
            inv_std = 1.0 / math.sqrt(var + self.eps)
            for j in range(len(x[i])):
                x[i][j] = self.gamma[j] * (x[i][j] - mean) * inv_std + self.beta[j]

        if is_vector:
            return x[0]
        return x


class Linear:
    def __init__(self, in_features, out_features):
        self.weight = [
            [random.uniform(-0.1, 0.1) for _ in range(in_features)]
            for _ in range(out_features)
        ]
        self.bias = [0.0] * out_features

    def forward(self, x):
        if isinstance(x[0], list):
            out = []
            for vec in x:
                out_vec = []
                for j in range(len(self.weight)):
                    value = self.bias[j]
                    for k in range(len(vec)):
                        value += self.weight[j][k] * vec[k]
                    out_vec.append(value)
                out.append(out_vec)
            return out

        out_vec = []
        for j in range(len(self.weight)):
            value = self.bias[j]
            for k in range(len(x)):
                value += self.weight[j][k] * x[k]
            out_vec.append(value)
        return out_vec


if __name__ == "__main__":
    bert = MiniBERT(
        vocab_size=100,
        max_len=20,
        hidden_size=16,
        num_heads=2,
        num_layers=2,
        intermediate_size=32,
    )

    input_ids = [
        [5, 12, 8, 2],
        [1, 3, 9, 7],
    ]
    segment_ids = [[0, 0, 0, 0], [0, 0, 0, 0]]
    attention_mask = [[1, 1, 1, 1], [1, 1, 1, 0]]

    logits = bert.forward(input_ids, segment_ids, attention_mask)

    print("Kich thuoc logits:", len(logits), "x", len(logits[0]), "x", len(logits[0][0]))
    print("Logits cho cau 1, token dau tien (5 chieu dau):", logits[0][0][:5])
