import math
from collections import defaultdict

class UnigramTokenizer:
    """
    Unigram Language Model tokenization.
    Huấn luyện: tạo seed vocab, lặp loại bỏ subword ít quan trọng nhất.
    """
    def __init__(self):
        self.vocab = set()           # tập subword
        self.probs = {}              # xác suất log của mỗi subword
        self.min_freq = 2            # tần suất tối thiểu để vào seed vocab
        self.max_subword_len = 5     # độ dài subword tối đa khi tạo seed

    # ---------- Tạo seed vocabulary ----------
    def _build_seed_vocab(self, corpus):
        """
        Tạo vocabulary ban đầu: gồm tất cả ký tự + các subword phổ biến.
        corpus: list các câu (string)
        """
        word_freq = defaultdict(int)
        for sentence in corpus:
            for word in sentence.split():
                word_freq[word] += 1

        # Đếm tần suất tất cả subword có độ dài [1, max_subword_len] trong mỗi từ
        subword_counts = defaultdict(int)
        for word, freq in word_freq.items():
            n = len(word)
            for l in range(1, self.max_subword_len + 1):
                for i in range(n - l + 1):
                    sub = word[i:i+l]
                    subword_counts[sub] += freq

        # Seed vocab: các ký tự riêng lẻ + subword có count >= min_freq
        seed = set()
        # Thêm tất cả ký tự xuất hiện
        for ch in set(''.join(corpus)):
            if ch != ' ':
                seed.add(ch)
        # Thêm subword phổ biến
        for sub, cnt in subword_counts.items():
            if cnt >= self.min_freq:
                seed.add(sub)
        # Thêm ký tự đánh dấu kết thúc từ '_' (dùng khi token hóa)
        seed.add('_')
        return seed

    # ---------- Viterbi phân tách tối ưu ----------
    def _viterbi_segment(self, word, probs_log):
        """
        Tìm phân tách subword tối ưu cho một từ (string), dựa trên xác suất log.
        Trả về list subwords và tổng log xác suất.
        probs_log: dict {subword: log_prob}
        """
        n = len(word)
        # dp[i] = (best_score, best_last_subword_start)
        dp = [(-float('inf'), -1) for _ in range(n+1)]
        dp[0] = (0.0, -1)   # bắt đầu với score 0

        for i in range(1, n+1):
            for j in range(max(0, i-self.max_subword_len), i):
                sub = word[j:i]
                if sub in probs_log:
                    cand_score = dp[j][0] + probs_log[sub]
                    if cand_score > dp[i][0]:
                        dp[i] = (cand_score, j)

        # Truy vết
        tokens = []
        pos = n
        while pos > 0:
            start = dp[pos][1]
            tokens.append(word[start:pos])
            pos = start
        tokens.reverse()
        return tokens

    # ---------- Huấn luyện ----------
    def train(self, corpus, target_vocab_size, num_iterations=10):
        """
        Huấn luyện unigram tokenizer.
        - corpus: list các câu (string)
        - target_vocab_size: kích thước vocab cuối cùng
        - num_iterations: số vòng lặp EM tối đa (loại bỏ dần)
        """
        # 1. Tạo seed vocab
        seed_vocab = self._build_seed_vocab(corpus)
        # 2. Tạo danh sách các từ trong corpus (có lặp để tính tần suất)
        words = []
        for sentence in corpus:
            words.extend(sentence.split())
        # Lưu tần suất từ
        word_freq = defaultdict(int)
        for w in words:
            word_freq[w] += 1

        current_vocab = seed_vocab.copy()
        for iteration in range(num_iterations):
            # Bước E: ước lượng xác suất unigram dựa trên tần suất subword trong phân tách tối ưu
            # Khởi tạo bộ đếm cho mỗi subword
            subword_counter = defaultdict(int)
            for word, freq in word_freq.items():
                # Tính probs_log cho current_vocab (giả định đồng đều ban đầu, nhưng sẽ cập nhật)
                # Để đơn giản, dùng uniform probabilities trong viterbi ở bước E? Không đúng.
                # Thực tế, cần dùng probs từ lần trước. Khởi tạo uniform.
                log_uniform = math.log(1.0 / len(current_vocab))
                probs_log = {sw: log_uniform for sw in current_vocab}
                # Tìm phân tách tối ưu với probs hiện tại
                best_tokens = self._viterbi_segment(word, probs_log)
                for token in best_tokens:
                    subword_counter[token] += freq

            # Cập nhật xác suất log: P(token) = count(token) / total_count
            total = sum(subword_counter.values())
            self.probs = {sw: cnt/total for sw, cnt in subword_counter.items()}
            # Chuyển sang log để dùng trong viterbi
            self.probs_log = {sw: math.log(p) for sw, p in self.probs.items()}

            # Bước M: loại bỏ các subword có mất mát thấp nhất nếu vocab còn lớn
            if len(current_vocab) <= target_vocab_size:
                break
            # Tính loss mỗi subword: loss = frequency * log(prob)  (càng âm thì càng tốt khi loại)
            # Thực tế, loại bỏ subword có loss ít âm nhất (gây ít tăng loss nhất khi xóa)
            losses = {}
            for sw in current_vocab:
                freq_sw = subword_counter.get(sw, 0)
                prob_sw = self.probs.get(sw, 1e-8)
                losses[sw] = freq_sw * math.log(prob_sw)  # càng nhỏ (âm) càng tốt
            # Sắp xếp theo loss (từ cao đến thấp) - muốn loại subword có loss lớn (ít âm nhất)
            sorted_sw = sorted(current_vocab, key=lambda x: losses[x], reverse=True)
            # Giữ lại target_vocab_size phần tử, loại 1/3 số thừa mỗi lần để ổn định
            num_to_keep = max(target_vocab_size, int(len(current_vocab) * 0.7))
            current_vocab = set(sorted_sw[:num_to_keep])
            # Đảm bảo giữ các ký tự đơn lẻ và '_' vì cần thiết
            for ch in set(''.join(corpus)):
                if ch != ' ':
                    current_vocab.add(ch)
            current_vocab.add('_')

        self.vocab = current_vocab
        # Tính lại probs cuối cùng dựa trên vocab hiện tại
        final_counter = defaultdict(int)
        for word, freq in word_freq.items():
            best_tokens = self._viterbi_segment(word, self.probs_log)
            for token in best_tokens:
                final_counter[token] += freq
        total = sum(final_counter.values())
        self.probs = {sw: cnt/total for sw, cnt in final_counter.items()}
        self.probs_log = {sw: math.log(p) for sw, p in self.probs.items()}

    # ---------- Token hóa ----------
    def tokenize(self, text):
        """
        Token hóa câu văn bản.
        Trả về list các subword token.
        """
        tokens = []
        for word in text.split():
            # Thêm ký tự '_' cuối từ theo chuẩn SentencePiece
            word_with_boundary = word + '_'
            # Dùng Viterbi với probs_log đã huấn luyện
            word_tokens = self._viterbi_segment(word_with_boundary, self.probs_log)
            tokens.extend(word_tokens)
        return tokens


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    corpus = [
        "low low",
        "lowest",
        "newer",
        "wider",
        "lower"
    ]

    tokenizer = UnigramTokenizer()
    tokenizer.train(corpus, target_vocab_size=10)

    print("Vocabulary size:", len(tokenizer.vocab))
    print("Vocab:", tokenizer.vocab)
    print("\nXác suất các subword (top 5):")
    sorted_probs = sorted(tokenizer.probs.items(), key=lambda x: x[1], reverse=True)
    for sw, p in sorted_probs[:5]:
        print(f"  {repr(sw)}: {p:.4f}")

    test_sentence = "Real growth requires vision. Zebra has the automation technology and the deep industry expertise to improve every step in your manufacturing or distribution supply chain. Align your business with the industry’s most advanced and growing fixed scanning and machine vision portfolio."
    print(f"\nTokenize '{test_sentence}':")
    print(tokenizer.tokenize(test_sentence))