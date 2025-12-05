# Spam/Ham Classification System

## Монгол хэлний сэтгэгдэл ангилагч

Энэхүү систем нь Facebook-ийн монгол хэлний сэтгэгдлүүдийг spam эсвэл ham (хэвийн) гэж ангилдаг.

## Бүтэц

```
spam_classifier_app/
├── main.py                    # Үндсэн консол application
├── data_loader.py             # Өгөгдөл ачаалах модуль
├── naive_bayes_model.py       # Multinomial Naive Bayes модель
├── decision_tree_model.py     # Decision Tree модель
├── model_evaluator.py         # Моделийг сургах болон үнэлэх
├── visualizations.py          # Визуализацийн модуль
├── fb_comment.csv             # Өгөгдлийн файл (CSV)
├── fb_comment.xlsx            # Өгөгдлийн файл (Excel)
└── README.md                  # Энэ файл
```

## Шаардлагатай сангууд

```bash
pip install pandas numpy scikit-learn seaborn matplotlib openpyxl
```

## Хэрхэн ажиллуулах

```bash
python main.py
```

## Боломжит функцүүд

### 1. Өгөгдөл ачаалах

- Google Sheets URL-ээс (default)
- CSV файлаас
- XLSX файлаас

### 2. Naive Bayes модель сургах

- **Текстийн эх үүсвэр сонгох**:
  - Raw comment (Анхны сэтгэгдэл)
  - Transliterated comment (Цэвэрлэсэн сэтгэгдэл)
  - Both (Хоёулаа нэгтгэсэн)
- Laplace smoothing (α) параметр тохируулах
- N-gram төрөл сонгох (unigram, bigram, trigram)
- Train/test хуваалт тохируулах

### 3. Decision Tree модель сургах

- Maximum depth тохируулах
- Train/test хуваалт тохируулах
- Tree бүтцийг харах

### 4. Моделийг үнэлэх

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score

### 5. Өөрийн сэтгэгдэл ангилуулах

- **Naive Bayes**: Текст оруулаад шууд ангилах
- **Decision Tree**: Шинжүүд (features) оруулж ангилах

### 6. Визуализаци харах

- Сэтгэгдлийн урт тархалт
- Emoji тоо тархалт
- Үсгийн төрөл (Кирилл/Латин/Бусад)
- Binary features хувь тархалт
- Feature=1 үед spam/ham тархалт
- Correlation matrix

## Жишээ

### Naive Bayes ашиглах:

1. Өгөгдөл ачаалах (Сонголт 1)
2. Naive Bayes сургах (Сонголт 2)
3. Моделийг үнэлэх (Сонголт 4)
4. Өөрийн сэтгэгдэл ангилуулах (Сонголт 5)

### Decision Tree ашиглах:

1. Өгөгдөл ачаалах (Сонголт 1)
2. Decision Tree сургах (Сонголт 3)
3. Моделийг үнэлэх (Сонголт 4)
4. Өөрийн сэтгэгдэл ангилуулах (Сонголт 5)

## Өгөгдлийн формат

### Naive Bayes-д шаардлагатай:

- `label`: spam эсвэл ham
- `Raw comment`: Сэтгэгдлийн текст

### Decision Tree-д шаардлагатай:

- `label`: spam эсвэл ham
- Binary features:
  - Зураг агуулсан эсэх
  - Нэрээ нууцалсан эсэх
  - Монгол нэр эсэх
  - Кирил, латин биш тэмдэгт ашигласан эсэх
  - Email агуулсан эсэх
  - Link агуулсан эсэх
  - Утасны дугаар агуулсан эсэх
- Numeric features:
  - Зөв бичсэн хувь
  - Ашигласан үсэг
  - Emoji-ний тоо
  - Сэтгэгдлийн урт

## Техникийн дэлгэрэнгүй

### Naive Bayes Algorithm

- Multinomial Naive Bayes
- N-gram tokenization (unigram, bigram, trigram)
- Laplace smoothing

### Decision Tree Algorithm

- ID3 algorithm
- Information Gain / Entropy
- Тоон болон ангилалтай шинжүүдийг дэмждэг

## Зохиогч

МУИС-МТЭС Магадлал Статистик хичээлийн 6-р багийн оюутан:

- М.Ариунзаяа
- С.Буян-Эрдэнэ
- Б.Эрдэнэ-Очир
- О.Нямбаяр
