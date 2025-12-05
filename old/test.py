import pandas as pd
import numpy as np

# =========================================================
#   Туслах функцүүд
# =========================================================

def plurality_value(examples, target):
    """Ихэнх давтамжтай утгыг буцаана"""
    counts = examples[target].value_counts()
    return counts.idxmax()

def entropy(examples, target):
    """Энтропи тооцоолох"""
    values, counts = np.unique(examples[target], return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

def remainder(examples, attr, target, threshold=None):
    """Remainder тооцоолох (ангилалтай болон тоон шинжид)"""
    if threshold is None:
        # Ангилалтай шинж
        values, counts = np.unique(examples[attr], return_counts=True)
        total = len(examples)
        rem = 0.0
        for v, c in zip(values, counts):
            subset = examples[examples[attr] == v]
            rem += (c / total) * entropy(subset, target)
        return rem
    else:
        # Тоон шинж
        left = examples[examples[attr] <= threshold]
        right = examples[examples[attr] > threshold]
        total = len(examples)
        if len(left) == 0 or len(right) == 0:
            return float('inf')  # Хуваалт муу бол
        rem = (len(left)/total) * entropy(left, target) + (len(right)/total) * entropy(right, target)
        return rem

def information_gain(examples, attr, target):
    """Information gain тооцоолох"""
    base_entropy = entropy(examples, target)
    
    if np.issubdtype(examples[attr].dtype, np.number):
        # Тоон шинжид хамгийн сайн threshold олох
        values = sorted(examples[attr].unique())
        if len(values) < 2:
            return 0, None
        
        best_gain, best_threshold = -1, None
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            rem = remainder(examples, attr, target, threshold)
            if rem == float('inf'):
                continue
            gain = base_entropy - rem
            if gain > best_gain:
                best_gain, best_threshold = gain, threshold
        return best_gain, best_threshold
    else:
        # Ангилалтай шинж
        gain = base_entropy - remainder(examples, attr, target)
        return gain, None

# =========================================================
#   Decision Tree сургах
# =========================================================

def decision_tree_learning(examples, attributes, target, parent_examples=None, max_depth=10, current_depth=0):
    """Decision Tree сургах (ID3 алгоритм)"""
    if len(examples) == 0:
        return plurality_value(parent_examples, target)
    
    if len(np.unique(examples[target])) == 1:
        return np.unique(examples[target])[0]
    
    if len(attributes) == 0 or current_depth >= max_depth:
        return plurality_value(examples, target)
    
    # Бүх шинжийн information gain тооцоолох
    gains, thresholds = {}, {}
    for attr in attributes:
        gain, threshold = information_gain(examples, attr, target)
        gains[attr] = gain
        thresholds[attr] = threshold
    
    # Хамгийн их gain-тай шинжийг сонгох
    best_attr = max(gains, key=gains.get)
    
    if gains[best_attr] <= 0:  # Ашигтай салгалт үүсэхгүй бол
        return plurality_value(examples, target)
    
    best_threshold = thresholds[best_attr]
    tree = {best_attr: {}}
    remaining_attrs = [a for a in attributes if a != best_attr]
    
    if best_threshold is not None:
        # Тоон шинж: <= болон > гэж хуваана
        left = examples[examples[best_attr] <= best_threshold]
        right = examples[examples[best_attr] > best_threshold]
        
        tree[best_attr][f'<= {round(best_threshold, 2)}'] = decision_tree_learning(
            left, remaining_attrs, target, examples, max_depth, current_depth + 1
        )
        tree[best_attr][f'> {round(best_threshold, 2)}'] = decision_tree_learning(
            right, remaining_attrs, target, examples, max_depth, current_depth + 1
        )
    else:
        # Ангилалтай шинж: утга бүрээр салгана
        for v in np.unique(examples[best_attr]):
            exs = examples[examples[best_attr] == v]
            subtree = decision_tree_learning(
                exs, remaining_attrs, target, examples, max_depth, current_depth + 1
            )
            tree[best_attr][v] = subtree
    
    return tree

# =========================================================
#   Таамаглал хийх
# =========================================================

def predict(tree, sample):
    """Нэг дээжинд таамаглал хийх"""
    if not isinstance(tree, dict):
        return tree
    
    attr = next(iter(tree))
    value = sample[attr]
    
    for condition, branch in tree[attr].items():
        if condition.startswith('<='):
            threshold = float(condition.split('<= ')[1])
            if value <= threshold:
                return predict(branch, sample)
        elif condition.startswith('>'):
            threshold = float(condition.split('> ')[1])
            if value > threshold:
                return predict(branch, sample)
        elif str(value) == str(condition):
            return predict(branch, sample)
    
    return None

def predict_dataset(tree, df):
    """Бүхэл датасет дээр таамаглал хийх"""
    predictions = []
    for idx, row in df.iterrows():
        pred = predict(tree, row)
        predictions.append(pred)
    return predictions

# =========================================================
#   Tree хэвлэх
# =========================================================

def print_tree(tree, indent=""):
    """Decision Tree-г хүснэгт маягаар хэвлэх"""
    if not isinstance(tree, dict):
        print(indent + "→ " + str(tree))
        return
    
    for attr, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}[{attr} = {value}]")
            print_tree(subtree, indent + "   ")

# =========================================================
#   Үнэлгээний функцүүд
# =========================================================

def accuracy(y_true, y_pred):
    """Нарийвчлал тооцоолох"""
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred, labels):
    """Confusion matrix үүсгэх"""
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            matrix[label_to_idx[yt]][label_to_idx[yp]] += 1
    
    return matrix

def classification_report_custom(y_true, y_pred, labels):
    """Classification report үүсгэх"""
    cm = confusion_matrix(y_true, y_pred, labels)
    
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label:15} Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

# =========================================================
#   MAIN - Жишээ ашиглалт
# =========================================================

if __name__ == "__main__":
    # Өгөгдлийн сан уншиж авах
    CSV_PATH = "fb_comment.xlsx"
    df = pd.read_excel(CSV_PATH)
    
    print("Өгөгдлийн хэмжээ:", df.shape)
    print("\nБагануудын нэрс:")
    print(df.columns.tolist())
    
    # Label багана
    COL_LABEL = "label"
    df = df.dropna(subset=[COL_LABEL])
    
    # Текстийн багана боловсруулах
    COL_COMMENT_CLEAN = "Цэвэрлэсэн сэтгэгдэл"
    
    # Ашиглах шинжүүд
    BOOL_COLUMNS = [
        "Зураг агуулсан эсэх",
        "Нэрээ нууцалсан эсэх",
        "Монгол нэр эсэх",
        "Кирил, латин биш тэмдэгт ашигласан эсэх",
        "Email агуулсан эсэх",
        "Link агуулсан эсэх",
        "Утасны дугаар агуулсан эсэх"
    ]
    
    NUMERIC_COLUMNS = [
        "Зөв бичсэн хувь",
        "Ашигласан үсэг",
        "Emoji-ний тоо",
        "Сэтгэгдлийн урт"
    ]
    
    # Boolean утгуудыг 0/1 болгох
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({
                True: 1, False: 0, 
                "True": 1, "False": 0,
                "yes": 1, "no": 0,
                "Yes": 1, "No": 0
            }).fillna(0).astype(int)
        else:
            df[col] = 0
    
    # Тоон утгуудыг боловсруулах
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0
    
    # Label-г string болгох
    df[COL_LABEL] = df[COL_LABEL].astype(str).str.strip().str.lower()
    
    print("\nLabel-н төрлүүд:", df[COL_LABEL].unique())
    print("Label-н тархалт:")
    print(df[COL_LABEL].value_counts())
    
    # Train/Test хуваах (80/20)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[COL_LABEL])
    
    print(f"\nТренинг датасет: {len(train_df)} мөр")
    print(f"Тест датасет: {len(test_df)} мөр")
    
    # Decision Tree сургах
    target = COL_LABEL
    attributes = BOOL_COLUMNS + NUMERIC_COLUMNS
    
    print("\n" + "="*60)
    print("Decision Tree сургаж байна...")
    print("="*60)
    
    tree = decision_tree_learning(
        train_df[attributes + [target]], 
        attributes, 
        target,
        max_depth=8  # max_depth-г тохируулж болно
    )
    
    print("\n--- Decision Tree бүтэц ---")
    print_tree(tree)
    
    # Тест дээр таамаглал хийх
    print("\n" + "="*60)
    print("Тест датасет дээр үнэлгээ хийж байна...")
    print("="*60)
    
    y_test_true = test_df[target].tolist()
    y_test_pred = predict_dataset(tree, test_df[attributes])
    
    acc = accuracy(y_test_true, y_test_pred)
    print(f"\nНарийвчлал (Accuracy): {acc:.4f}")
    
    print("\nClassification Report:")
    labels = sorted(df[COL_LABEL].unique())
    classification_report_custom(y_test_true, y_test_pred, labels)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_true, y_test_pred, labels)
    print("              ", "  ".join(f"{l:10}" for l in labels))
    for i, label in enumerate(labels):
        print(f"{label:15}", "  ".join(f"{cm[i][j]:10}" for j in range(len(labels))))
    
    # Жишээ таамаглал - датанаас санамсаргүй сонгох
    print("\n" + "="*60)
    print("Жишээ таамаглалууд (тест датанаас санамсаргүй):")
    print("="*60)
    
    # Тест датанаас 5 мөр санамсаргүй сонгох
    sample_rows = test_df.sample(n=min(5, len(test_df)), random_state=42)
    
    for idx, (row_idx, row) in enumerate(sample_rows.iterrows(), 1):
        print(f"\n--- Жишээ #{idx} (index: {row_idx}) ---")
        
        # Бодит утга
        true_label = row[target]
        print(f"Бодит label: {true_label}")
        
        # Таамаглал хийх
        prediction = predict(tree, row[attributes])
        print(f"Таамаглал: {prediction}")
        print(f"Зөв эсэх: {'✓' if prediction == true_label else '✗'}")
        
        # Шинжүүдийг харуулах
        print("\nШинжүүд:")
        for attr in attributes:
            print(f"  {attr}: {row[attr]}")
        
        # Хэрэв текстийн багана байвал харуулах
        if COL_COMMENT_CLEAN in test_df.columns:
            comment_text = row[COL_COMMENT_CLEAN]
            if pd.notna(comment_text) and str(comment_text).strip():
                print(f"\nСэтгэгдэл: {str(comment_text)[:100]}...")  # Эхний 100 тэмдэгт