import math
import random
from collections import defaultdict

class Word2Vec:
    """
    Word2Vec Skip-gram with Negative Sampling.
    Huấn luyện embedding cho từ vựng dựa trên ngữ cảnh cục bộ.
    """
    def __init__(self, vector_size=100, window=5, neg_samples=5, learning_rate=0.05, epochs=5):
        self.vector_size = vector_size   # số chiều embedding
        self.window = window             # kích thước cửa sổ ngữ cảnh (mỗi bên)
        self.neg_samples = neg_samples   # số lượng negative samples
        self.lr = learning_rate          # tốc độ học
        self.epochs = epochs             # số lần duyệt toàn bộ corpus
        self.word2id = {}                # từ -> index
        self.id2word = {}                # index -> từ
        self.vocab_size = 0
        self.word_freq = defaultdict(int) # tần suất từ
        self.W1 = None                   # ma trận embedding từ (vocab_size x vector_size)
        self.W2 = None                   # ma trận embedding ngữ cảnh (vector_size x vocab_size)

    # ---------- Tiền xử lý ----------
    def _build_vocab(self, corpus):
        """
        Xây dựng từ điển từ corpus (list các câu, mỗi câu là list các từ).
        """
        freq = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
                freq[word] += 1
        # Gán id cho mỗi từ (có thể lọc bỏ từ quá hiếm, ở đây giữ tất cả)
        self.word2id = {}
        self.id2word = {}
        for idx, (word, count) in enumerate(freq.items()):
            self.word2id[word] = idx
            self.id2word[idx] = word
            self.word_freq[idx] = count
        self.vocab_size = len(self.word2id)

    # ---------- Tạo dữ liệu huấn luyện (ngữ cảnh, từ trung tâm, negative) ----------
    def _generate_training_pairs(self, corpus):
        """
        Sinh các cặp (ngữ cảnh, từ trung tâm) và negative samples.
        Trả về list các tuple: (center_idx, context_idx, negative_indices)
        """
        pairs = []
        # Tính phân phối cho negative sampling: P(word) ~ (count)^(3/4) chia tổng
        self.neg_table = []  # bảng để lấy mẫu nhanh
        total_pow = 0.0
        pow_freq = {}
        for w, cnt in self.word_freq.items():
            p = cnt ** 0.75
            pow_freq[w] = p
            total_pow += p
        # Tạo bảng 10000 phần tử để sampling
        table_size = 10000
        self.neg_table = [0] * table_size
        i = 0
        for w, p in pow_freq.items():
            count = int(p / total_pow * table_size)
            for _ in range(count):
                if i < table_size:
                    self.neg_table[i] = w
                    i += 1
        # Fill nốt nếu thiếu
        while i < table_size:
            self.neg_table[i] = random.randint(0, self.vocab_size-1)
            i += 1

        for sentence in corpus:
            sent_len = len(sentence)
            for center_pos, center_word in enumerate(sentence):
                center_idx = self.word2id[center_word]
                # Lấy ngữ cảnh trong cửa sổ window
                for offset in range(-self.window, self.window+1):
                    if offset == 0:
                        continue
                    context_pos = center_pos + offset
                    if 0 <= context_pos < sent_len:
                        context_word = sentence[context_pos]
                        context_idx = self.word2id[context_word]
                        # Lấy negative samples (khác với context hiện tại)
                        neg_idxs = []
                        while len(neg_idxs) < self.neg_samples:
                            neg_idx = self.neg_table[random.randint(0, len(self.neg_table)-1)]
                            if neg_idx != context_idx:
                                neg_idxs.append(neg_idx)
                        pairs.append((center_idx, context_idx, neg_idxs))
        return pairs

    # ---------- Hàm kích hoạt và đạo hàm ----------
    def _sigmoid(self, x):
        """Hàm sigmoid: 1/(1+exp(-x))"""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)

    # ---------- Huấn luyện ----------
    def train(self, corpus):
        """
        Huấn luyện Word2Vec.
        corpus: list các câu, mỗi câu là list các string (đã tokenize sẵn).
        """
        # Xây dựng từ điển
        self._build_vocab(corpus)
        print(f"Vocab size: {self.vocab_size}")

        # Khởi tạo ma trận embedding ngẫu nhiên nhỏ
        self.W1 = [[random.uniform(-0.5/self.vector_size, 0.5/self.vector_size)
                    for _ in range(self.vector_size)] for _ in range(self.vocab_size)]
        self.W2 = [[random.uniform(-0.5/self.vector_size, 0.5/self.vector_size)
                    for _ in range(self.vocab_size)] for _ in range(self.vector_size)]

        # Sinh cặp huấn luyện (chỉ 1 lần, không đổi qua epoch)
        training_pairs = self._generate_training_pairs(corpus)
        print(f"Total training pairs: {len(training_pairs)}")

        for epoch in range(self.epochs):
            total_loss = 0.0
            random.shuffle(training_pairs)
            for center_idx, context_idx, neg_idxs in training_pairs:
                # Forward: lấy vector center (W1[center]) và context/negative (W2[:, context] và W2[:, neg])
                center_vec = self.W1[center_idx]  # list length vector_size
                # Tính tích vô hướng cho context
                context_vec = self.W2[context_idx] # thực tế W2 là ma trận cột, lấy cột context
                # Dot product
                score = 0.0
                for d in range(self.vector_size):
                    score += center_vec[d] * context_vec[d]
                loss_pos = -math.log(self._sigmoid(score))
                total_loss += loss_pos
                # Gradient cho positive pair
                grad = self._sigmoid(score) - 1   # dL/d(score) = sigmoid - 1

                # Cập nhật W1 và W2 cho positive
                # Update W1[center]
                for d in range(self.vector_size):
                    self.W1[center_idx][d] -= self.lr * grad * context_vec[d]
                # Update W2[context]
                for d in range(self.vector_size):
                    self.W2[context_idx][d] -= self.lr * grad * center_vec[d]

                # Negative samples
                for neg_idx in neg_idxs:
                    neg_vec = self.W2[neg_idx]
                    score_neg = 0.0
                    for d in range(self.vector_size):
                        score_neg += center_vec[d] * neg_vec[d]
                    loss_neg = -math.log(self._sigmoid(-score_neg))
                    total_loss += loss_neg
                    grad_neg = self._sigmoid(score_neg)  # dL/d(score) = sigmoid
                    # Cập nhật
                    for d in range(self.vector_size):
                        self.W1[center_idx][d] -= self.lr * grad_neg * neg_vec[d]
                    for d in range(self.vector_size):
                        self.W2[neg_idx][d] -= self.lr * grad_neg * center_vec[d]

            print(f"Epoch {epoch+1}/{self.epochs}  Loss: {total_loss:.4f}")

    # ---------- Lấy embedding ----------
    def get_embedding(self, word):
        """Trả về vector (list) cho một từ."""
        if word not in self.word2id:
            return None
        idx = self.word2id[word]
        return self.W1[idx][:]

    def most_similar(self, word, topn=5):
        """Tìm topn từ gần nhất với word dựa trên cosine similarity."""
        if word not in self.word2id:
            return []
        idx = self.word2id[word]
        vec = self.W1[idx]
        # Tính cosine với tất cả các từ khác
        sims = []
        for other_idx, other_vec in enumerate(self.W1):
            if other_idx == idx:
                continue
            # Dot product
            dot = 0.0
            norm1 = 0.0
            norm2 = 0.0
            for d in range(self.vector_size):
                dot += vec[d] * other_vec[d]
                norm1 += vec[d]**2
                norm2 += other_vec[d]**2
            if norm1 == 0 or norm2 == 0:
                cos = -1
            else:
                cos = dot / (math.sqrt(norm1) * math.sqrt(norm2))
            sims.append((cos, self.id2word[other_idx]))
        sims.sort(reverse=True)
        return sims[:topn]


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Tạo corpus nhỏ (đã tokenize sẵn)
    corpus = [
        ["i", "like", "deep", "learning"],
        ["i", "like", "natural", "language", "processing"],
        ["deep", "learning", "is", "fun"],
        ["natural", "language", "processing", "is", "cool"],
        ["i", "am", "learning", "word", "embeddings"]
    ]

    w2v = Word2Vec(vector_size=10, window=2, neg_samples=3, learning_rate=0.05, epochs=10)
    w2v.train(corpus)

    # Xem embedding của từ "learning"
    emb = w2v.get_embedding("learning")
    print("\nEmbedding của 'learning':", emb[:5], "...")  # in 5 chiều đầu

    # Tìm từ tương tự
    print("\nTừ tương tự với 'learning':")
    for sim, word in w2v.most_similar("learning", topn=3):
        print(f"  {word}: {sim:.4f}")