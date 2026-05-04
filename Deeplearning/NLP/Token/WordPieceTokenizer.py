import math
from collections import defaultdict

class WordPieceTokenizer:
    def __init__(self):
        self.merges = []   # list of (pair, new_token)
        self.vocab = set()
    
    def _get_counts(self, words):
        """Đếm tần suất của mỗi token trong danh sách các từ (mỗi từ là list token)"""
        freq = defaultdict(int)
        for word in words:
            for token in word:
                freq[token] += 1
        return freq
    
    def _get_pair_counts(self, words):
        """Đếm tần suất các cặp (a,b) liền nhau"""
        pair_counts = defaultdict(int)
        for word in words:
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pair_counts[pair] += 1
        return pair_counts
    
    def _merge_pair(self, pair, new_token, words):
        """Thay thế tất cả cặp (pair) bằng new_token trong words"""
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
    
    def train(self, corpus, num_merges, smooth=1e-6):
        """
        Huấn luyện WordPiece.
        corpus: list các câu (string)
        num_merges: số lần merge
        smooth: tránh chia 0
        """
        # Khởi tạo words: mỗi từ trong corpus sau khi tách khoảng trắng, biểu diễn thành list ký tự + '_'
        words = []
        for sentence in corpus:
            for word in sentence.split():
                word_tokens = list(word) + ['_']  # '_' ký tự kết thúc từ
                words.append(word_tokens)
        
        # Khởi tạo vocab ban đầu: tất cả token xuất hiện
        freq = self._get_counts(words)
        self.vocab = set(freq.keys())
        
        for step in range(num_merges):
            pair_counts = self._get_pair_counts(words)
            if not pair_counts:
                break
            
            # Tính score cho mỗi cặp
            best_pair = None
            best_score = -float('inf')
            for (a,b), cnt_ab in pair_counts.items():
                cnt_a = freq.get(a, 0)
                cnt_b = freq.get(b, 0)
                # WordPiece thường dùng likelihood: score = cnt_ab / (cnt_a * cnt_b)
                # Hoặc log version
                score = cnt_ab / (cnt_a * cnt_b + smooth)
                if score > best_score:
                    best_score = score
                    best_pair = (a,b)
            
            if best_pair is None:
                break
            
            # Merge best_pair
            new_token = best_pair[0] + best_pair[1]
            self.merges.append((best_pair, new_token))
            words = self._merge_pair(best_pair, new_token, words)
            # Cập nhật freq sau merge
            freq = self._get_counts(words)
            self.vocab.add(new_token)
        
        # Thêm tất cả token hiện tại vào vocab
        self.vocab.update(freq.keys())
    
    def _apply_merges(self, word_tokens):
        """Áp dụng các merge đã học lên một từ dạng token list (theo thứ tự merge)"""
        current = word_tokens
        for (a,b), new_token in self.merges:
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
        """Token hóa câu văn bản thành list subword token"""
        tokens = []
        for word in text.split():
            # Chuyển thành list ký tự + '_'
            word_tokens = list(word) + ['_']
            subwords = self._apply_merges(word_tokens)
            tokens.extend(subwords)
        return tokens


# Ví dụ minh họa
if __name__ == "__main__":
    corpus = [
        "low low",
        "lowest",
        "newer",
        "wider",
        "lower"
    ]
    
    tokenizer = WordPieceTokenizer()
    tokenizer.train(corpus, num_merges=10)
    
    print("Các merge (theo thứ tự):")
    for (a,b), new_token in tokenizer.merges:
        print(f"{repr(a)} + {repr(b)} -> {repr(new_token)}")
    
    print("\nVocabulary:", tokenizer.vocab)
    
    test = "lower newer low"
    print(f"\nTokenize '{test}': {tokenizer.tokenize(test)}")