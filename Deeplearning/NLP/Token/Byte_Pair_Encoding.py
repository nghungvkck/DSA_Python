from collections import defaultdict

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer (subword).
    Huấn luyện dựa trên tần suất cặp ký tự, sau đó tokenize câu mới.
    """
    def __init__(self):
        self.merges = []          # lưu các cặp đã merge theo thứ tự: [(pair1, new_token), ...]
        self.vocab = set()        # tập token sau khi huấn luyện

    def _get_stats(self, words):
        """
        Đếm tần suất xuất hiện của các cặp (pair) trong tất cả các từ.
        words: list of list of tokens (mỗi từ là một list các token hiện tại, thường bắt đầu là ký tự)
        """
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] += 1
        return pairs

    def _merge_pair(self, pair, new_token, words):
        """
        Hợp nhất một cặp (a,b) thành new_token trong tất cả các từ.
        words: list of list of tokens -> cập nhật trực tiếp
        """
        a, b = pair
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == a and word[i+1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def train(self, corpus, num_merges):
        """
        Huấn luyện BPE trên corpus.
        corpus: list các câu (string). Ví dụ: ["low low", "lowest", "newer"]
        num_merges: số lần merge cặp có tần suất cao nhất.
        """
        # Bước 0: tách từ, thêm ký tự kết thúc từ '_' và chia thành ký tự
        words = []
        for sentence in corpus:
            for word in sentence.split():
                # Biểu diễn từ dưới dạng list các ký tự, thêm '_' cuối để đánh dấu hết từ
                tokenized_word = list(word) + ['_']
                words.append(tokenized_word)

        # Lưu lại state ban đầu cho việc token hóa sau (dùng để biết nếu gặp từ chưa thấy)
        self.initial_words = [list(word) + ['_'] for word in set(' '.join(corpus).split())]  # mẫu

        # Lặp merge
        for i in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break
            # Chọn cặp xuất hiện nhiều nhất
            best_pair = max(pairs, key=pairs.get)
            new_token = ''.join(best_pair)   # tạo token mới bằng cách ghép 2 token cũ
            self.merges.append((best_pair, new_token))
            # Cập nhật words
            words = self._merge_pair(best_pair, new_token, words)
            self.vocab.add(new_token)

        # Thêm các ký tự đơn lẻ vào vocab (để token hóa được)
        for word in words:
            for token in word:
                self.vocab.add(token)
        # Thêm ký tự '_' nếu chưa có
        self.vocab.add('_')

    def _apply_merges(self, word_tokens):
        """
        Áp dụng các merge đã học lên một từ dạng list token (VD: ['l','o','w','_']).
        Trả về list token mới.
        """
        current = word_tokens
        for (a,b), new_token in self.merges:
            # Hợp nhất cặp (a,b) trong current
            new_current = []
            i = 0
            while i < len(current):
                if i < len(current)-1 and current[i] == a and current[i+1] == b:
                    new_current.append(new_token)
                    i += 2
                else:
                    new_current.append(current[i])
                    i += 1
            current = new_current
        return current

    def tokenize(self, text):
        """
        Token hóa một câu văn bản (string) thành list các subword token.
        """
        tokens = []
        for word in text.split():
            # Chuyển từ thành list ký tự + '_'
            word_tokens = list(word) + ['_']
            # Áp dụng các merge
            subwords = self._apply_merges(word_tokens)
            tokens.extend(subwords)
        return tokens


# ------------------- VÍ DỤ MINH HỌA -------------------
if __name__ == "__main__":
    # Corpus mẫu
    corpus = [
        "low low",
        "lowest",
        "newer",
        "wider",
        "lower"
    ]

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, num_merges=10)   # merge 10 lần

    print("Các merge đã học:")
    for (a,b), new_tok in tokenizer.merges:
        print(f"Merge {repr(a)} + {repr(b)} -> {repr(new_tok)}")

    print("\nToken vocab:", tokenizer.vocab)

    # Kiểm tra trên câu mới
    test_sentence = "lower newer low"
    print(f"\nCâu: '{test_sentence}'")
    result = tokenizer.tokenize(test_sentence)
    print("Token hóa:", result)