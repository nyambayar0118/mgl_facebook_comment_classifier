from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import math
import ssl
import seaborn as sns
import matplotlib.pyplot as plt

# HTTPS сертификатын алдааг алгасах (Google Sheets-ээс уншихад хэрэгтэй)
ssl._create_default_https_context = ssl._create_unverified_context


# =========================
# 1. N-gram болгож tokenize хийх функц
# =========================
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


# =========================
# 2. Өөрийн бичсэн Multinomial Naive Bayes класс
# =========================
class MyMultinomialNB:
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


# =========================
# 3. Өгөгдөл унших (Google Sheets -> CSV)
# =========================
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQC0C_5Sim0DhMAZbxRNxpFxBajpiHEGKeGWtSLLgbUWizbdrtoNGvv-x0P1phRr7DT6J5rCVyGzaYF/pub?output=csv'

df = pd.read_csv(
    url,
    engine='python',
    on_bad_lines='skip'   # эвдэрхий мөр байвал алгасаад явна
)

# df_raw нь бүх баганатай үндсэн дата (визуализацид ашиглах)
df_raw = df.copy()

# Эхний мөрийг header болгож, iloc[0]-ийг баганын нэрэнд ашиглаж байна
df_raw.columns = df_raw.iloc[0].astype(str).str.strip()
df_raw = df_raw.iloc[1:]  # эхний мөрийг өгөгдлөөс хасна

# label-уудыг жижиг үсгээр, зайгүй болгож цэвэрлэх
df_raw['label'] = df_raw['label'].str.strip().str.lower()

# =========================
# 4. Классфикаторын хувьд хэрэгтэй багануудыг тусад нь авах
# =========================
df_clf = df_raw[['label', 'Raw comment']].dropna()
df_clf = df_clf.rename(columns={'Raw comment': 'raw_comment'})

X = df_clf['raw_comment']
y = df_clf['label']

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# =========================
# 5. Naive Bayes model сургах (unigram + bigram)
# =========================
model = MyMultinomialNB(alpha=1.0, ngram_range=(1, 2))
model.fit(X_train, y_train)

# =========================
# 6. Тест өгөгдөл дээр таамаглал
# =========================
y_pred = model.predict(X_test)

# Жишээ текстүүдийг ангилуулах
print("Prediction:", model.predict(["Таалагдсан шүү aaa баярлалаа  "]))
print("Prediction:", model.predict(["Click this link to win iPhone"]))

# Үр дүн (accuracy, confusion matrix)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
labels = sorted(df_clf['label'].unique())
df_cm = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in labels],
    columns=[f"Pred {c}" for c in labels]
)
print("\nConfusion matrix:\n", df_cm)


# =========================
# 7. Визуализаци (comment length, emoji count, script type)
# =========================

# ----- 7.1. Comment length histogram -----
# 'Сэтгэгдлийн урт' баганыг тоон төрөлд хөрвүүлж, NaN-уудыг хаяна
df_raw['Сэтгэгдлийн урт'] = pd.to_numeric(df_raw['Сэтгэгдлийн урт'], errors='coerce')
# Зөвхөн label + урт байгаа мөрүүдийг авна
df_len = df_raw[['label', 'Сэтгэгдлийн урт']].dropna()

plt.figure(figsize=(12, 6))

sns.histplot(
    data=df_len,
    x='Сэтгэгдлийн урт',
    hue='label',          # spam / ham-ийг өнгөөр ялгана
    bins=10,
    multiple='dodge',     # нэг bin дээр 2 багана (spam, ham) зэрэгцүүлж зурна
    stat='count',
    palette={'spam': '#ffcccc', 'ham': '#ccffcc'}  # light red for spam, light green for ham
)
plt.title("Comment length distribution by class (spam vs ham)")
plt.xlabel("Сэтгэгдлийн урт (тэмдэгтүүдийн тоо)")
plt.ylabel("Давтамж")

# X тэнхлэгийн tick-үүдийг цэгцтэй болгоё
max_len = int(df_len['Сэтгэгдлийн урт'].max())
step = 200
plt.xticks(range(0, max_len + step, step), rotation=45)

plt.tight_layout()
plt.show()

# ----- 7.2. Emoji-ний тоо дээр histogram -----
plt.figure(figsize=(12, 5))

# Convert emoji count to numeric
df_raw['Emoji-ний тоо'] = pd.to_numeric(df_raw['Emoji-ний тоо'], errors='coerce').fillna(0).astype(int)

# Clip extreme outlier
df_plot = df_raw.copy()
df_plot['Emoji-ний тоо'] = df_plot['Emoji-ний тоо'].clip(upper=10)

sns.histplot(
    data=df_plot,
    x='Emoji-ний тоо',
    hue='label',
    bins=10,
    multiple='dodge',
    stat='count',
    palette={'spam': '#ffcccc', 'ham': '#ccffcc'}  # light red for spam, light green for ham
)

plt.title("Emoji count distribution by class (spam vs ham)")
plt.xlabel("Emoji-ний тоо (≥10 → 10 гэж нэгтгэсэн)")
plt.ylabel("Давтамж")
# Set x ticks 0–10
plt.xticks(range(0, 11))
plt.tight_layout()
plt.show()

# ----- 7.3. Голдуу ашигласан үсэг (кирилл/латин/бусад) pie chart -----
# 'Голдуу ашигласан үсэг' баганад хэдэн төрөл байгааг тоолно
plt.figure(figsize=(10, 5))

labels = ['spam', 'ham']
# Define colors for different script types, using lighter/darker shades based on spam/ham
colors_spam = ['#ff9999', '#ffcccc', '#ff6666', '#ffb3b3']  # shades of red for spam
colors_ham = ['#99ff99', '#ccffcc', '#66ff66', '#b3ffb3']   # shades of green for ham
colors_map = {'spam': colors_spam, 'ham': colors_ham}

for i, lbl in enumerate(labels):
    plt.subplot(1, 2, i+1)
    
    subset = df_raw[df_raw['label'] == lbl]
    script_counts = subset['Голдуу ашигласан үсэг'].value_counts()

    plt.pie(
        script_counts.values,
        labels=script_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_map[lbl][:len(script_counts)]
    )
    plt.title(f"Голдуу ашигласан үсэг — {lbl}")

plt.tight_layout()
plt.show()

# ===============================================================
# 7.4 Improved Binary Feature Analysis (Balanced & Non-overlapping)
# ===============================================================

import numpy as np

# Column rename mapping (if needed)
rename_map = {
    'Зураг агуулсан эсэх': 'contains_photo',
    'Нэрээ нууцалсан эсэх': 'is_anonymous',
    'Монгол нэр эсэх': 'is_mongolian_name',
    'Кирил, латин биш тэмдэгт ашигласан эсэх': 'has_weird_chars',
    'Email агуулсан эсэх': 'contains_email',
    'Link агуулсан эсэх': 'contains_link',
    'Утасны дугаар агуулсан эсэх': 'contains_number'
}
df_raw = df_raw.rename(columns=rename_map)

binary_features = [
    'contains_photo',
    'is_anonymous',
    'is_mongolian_name',
    'has_weird_chars',
    'contains_email',
    'contains_link',
    'contains_number'
]

# Convert to numeric 0/1 where needed
for col in binary_features:
    df_raw[col] = (
        df_raw[col].astype(str).str.strip().str.lower()
        .replace({'yes':1, 'no':0, 'true':1, 'false':0,
                  'тийм':1, 'үгүй':0, '1':1, '0':0})
    )
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0).astype(int)




# ===============================================================
# 7.4.1 Percent distribution (3 at a time, with count + percentage labels)
# ===============================================================

chunk_size = 3
total_rows = len(df_raw)

for start in range(0, len(binary_features), chunk_size):
    end = start + chunk_size
    chunk = binary_features[start:end]

    plt.figure(figsize=(18, 6))

    for i, col in enumerate(chunk):
        plt.subplot(1, len(chunk), i+1)

        counts = df_raw[col].value_counts().sort_index()               
        pct = (counts / total_rows * 100).round(2)

        ax = sns.barplot(
            x=counts.index.astype(str),
            y=pct.values,
            palette=["#ffcccc", "#ccffcc"]  # 0=red, 1=green
        )

        plt.title(f"{col} — % Distribution", fontsize=12)
        plt.xlabel("0 = No, 1 = Yes")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)

        for index, value in enumerate(pct.values):
            count_value = counts.iloc[index]
            ax.text(
                index,
                value + 2,
                f"{value}%\n({count_value})",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.show()




# ===============================================================
# 7.4.2 Reverse Percentage:
# Out of comments where feature == 1, what % are spam vs ham?
# ===============================================================

chunk_size = 3

for start in range(0, len(binary_features), chunk_size):
    end = start + chunk_size
    chunk = binary_features[start:end]

    plt.figure(figsize=(18, 6))

    for i, col in enumerate(chunk):
        plt.subplot(1, len(chunk), i+1)

        # Filter only rows where the feature is 1
        df_feature = df_raw[df_raw[col] == 1]

        total_with_feature = len(df_feature)

        if total_with_feature == 0:
            # Avoid division by zero
            plt.title(f"{col} — No rows with value 1")
            plt.bar(["spam", "ham"], [0, 0])
            continue

        # Count spam and ham among rows with feature == 1
        spam_count = (df_feature["label"] == "spam").sum()
        ham_count  = (df_feature["label"] == "ham").sum()

        spam_pct = round(spam_count / total_with_feature * 100, 2)
        ham_pct  = round(ham_count  / total_with_feature * 100, 2)

        values = [spam_pct, ham_pct]
        labels = ["spam", "ham"]

        ax = sns.barplot(
            x=labels,
            y=values,
            palette={'spam': '#ffcccc', 'ham': '#ccffcc'}  # light red for spam, light green for ham
        )

        plt.title(f"{col} — Among feature=1", fontsize=12)
        plt.ylabel("% of comments (feature=1)")
        plt.ylim(0, 100)

        # Annotate bars
        for idx, val in enumerate(values):
            count_val = spam_count if idx == 0 else ham_count
            ax.text(
                idx,
                val + 2,
                f"{val}%\n({count_val} comments)",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.show()




# ===============================================================
# 7.4.3 Updated heatmap
# ===============================================================

heatmap_cols = ['Сэтгэгдлийн урт', 'Emoji-ний тоо'] + binary_features
plt.figure(figsize=(14, 10))
sns.heatmap(
    df_raw[heatmap_cols].corr(),
    annot=True, fmt=".2f",
    cmap="coolwarm", linewidths=0.5
)
plt.title("Correlation Matrix for All Numerical & Binary Features")
plt.tight_layout()
plt.show()