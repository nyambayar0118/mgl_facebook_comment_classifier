"""
naive_bayes_model.py - Multinomial Naive Bayes модель
"""

from collections import Counter
import math


def ngram_tokenize(text, ngram_range=(2, 2)):
    """
    Өгөгдсөн text-ийг үгэнд хуваагаад ngram_range=(min_n, max_n)
    тохиргооны дагуу unigram/bigram/trigram гэх мэт жагсаалт болгож буцаана.
    Жишээ:
        text = "hello world nice day", ngram_range=(1,2)
        -> ["hello", "world", "nice", "day",
            "hello world", "world nice", "nice day"]
    """
    words = text.lower().split()
    n_min, n_max = ngram_range
    ngrams = []

    for n in range(n_min, n_max + 1):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            ngrams.append(ngram)

    return ngrams


class MyMultinomialNB:
    """Өөрийн бичсэн Multinomial Naive Bayes класс"""
    
    def __init__(self, alpha=1.0, ngram_range=(2, 2)):
        # Laplace тэгшитгэлд ашиглах α параметр (smoothing)
        self.alpha = alpha
        # Ямар хэмжээний n-gram ашиглах (unigram, bigram, ... )
        self.ngram_range = ngram_range

        # log P(c) хадгалах dict
        self.class_log_prior_ = {}
        # log P(ngram | c) хадгалах dict
        self.feature_log_probs_ = {}
        # Нийт vocabulary (бүх ангид гарсан бүх n-gram)
        self.vocabulary_ = set()

    def fit(self, X, y):
        """
        X: текстүүдийн жагсаалт
        y: ангиллын шошго (spam/ham)
        """
        # Ангилал тус бүрийн document-ийн тоо (N_c)
        class_counts = Counter(y)
        total_docs = len(y)

        # ----- 2.1. Приор магадлал log P(c) -----
        for c in class_counts:
            self.class_log_prior_[c] = math.log(class_counts[c] / total_docs)

        # ----- 2.2. Ангилал тус бүрийн n-gram-ийн давтамж тоолох -----
        class_ngram_counts = {c: Counter() for c in class_counts}  # N_{jc}
        total_ngrams = {c: 0 for c in class_counts}                # N_c (all n-grams)

        for text, label in zip(X, y):
            ngrams = ngram_tokenize(text, self.ngram_range)
            self.vocabulary_.update(ngrams)
            class_ngram_counts[label].update(ngrams)
            total_ngrams[label] += len(ngrams)

        vocab_size = len(self.vocabulary_)

        # ----- 2.3. Нөхцөлт магадлал θ_{jc} = P(ngram_j | class_c) -----
        # θ_{jc} = (N_{jc} + α) / (N_c + α * V)
        for c in class_counts:
            self.feature_log_probs_[c] = {}
            for ng in self.vocabulary_:
                count = class_ngram_counts[c][ng]
                prob = (count + self.alpha) / (total_ngrams[c] + self.alpha * vocab_size)
                # log P(ngram | c)
                self.feature_log_probs_[c][ng] = math.log(prob)

    def predict(self, X):
        """
        X: текстүүдийн жагсаалт
        Буцаах: ангиллын жагсаалт (хамгийн их posterior магадлалтай ангилал)
        """
        preds = []
        for text in X:
            ngrams = ngram_tokenize(text, self.ngram_range)
            class_scores = {}

            for c in self.class_log_prior_:
                # Эхлээд score = log P(c)
                score = self.class_log_prior_[c]

                # Дараа нь ∑ log P(ngram | c) нэмж өгнө
                for ng in ngrams:
                    if ng in self.vocabulary_:
                        score += self.feature_log_probs_[c][ng]

                class_scores[c] = score

            # argmax_c score(c)
            best_class = max(class_scores, key=class_scores.get)
            preds.append(best_class)

        return preds
    
    def predict_single(self, text):
        """Нэг текст ангилах"""
        return self.predict([text])[0]
    
    def save_vocabulary(self, filename='vocabulary.txt'):
        """Vocabulary-г файл руу хадгалах"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("N-GRAM VOCABULARY\n")
            f.write("="*70 + "\n")
            f.write(f"N-gram range: {self.ngram_range}\n")
            f.write(f"Total vocabulary size: {len(self.vocabulary_)}\n")
            f.write(f"Laplace smoothing (α): {self.alpha}\n")
            f.write("="*70 + "\n\n")
            
            # N-gram бүрийг бичих
            f.write("ALL N-GRAMS:\n")
            f.write("-"*70 + "\n")
            for i, ngram in enumerate(sorted(self.vocabulary_), 1):
                f.write(f"{i:5d}. {ngram}\n")
            
            # Class тус бүрийн top n-grams
            f.write("\n" + "="*70 + "\n")
            f.write("TOP N-GRAMS BY CLASS (by log probability)\n")
            f.write("="*70 + "\n\n")
            
            for class_label in sorted(self.feature_log_probs_.keys()):
                f.write(f"\n--- CLASS: {class_label.upper()} ---\n")
                f.write("-"*70 + "\n")
                
                # Log probability-оор эрэмбэлэх
                ngrams_with_probs = [
                    (ng, prob) for ng, prob in self.feature_log_probs_[class_label].items()
                ]
                ngrams_with_probs.sort(key=lambda x: x[1], reverse=True)
                
                # Эхний 50-ыг харуулах
                f.write(f"Top 50 most probable n-grams for {class_label}:\n\n")
                for i, (ngram, log_prob) in enumerate(ngrams_with_probs[:50], 1):
                    f.write(f"{i:3d}. {ngram:40s} (log_prob: {log_prob:.6f})\n")
        
        print(f"✅ Vocabulary хадгалагдлаа: {filename}")
        return filename
