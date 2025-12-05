"""
model_evaluator.py - Моделийг сургах болон үнэлэх модуль
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np


class ModelEvaluator:
    """Моделийг сургах болон үнэлэх класс"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def train_naive_bayes(self, X, y, alpha=1.0, ngram_range=(1, 2), test_size=0.3):
        """Naive Bayes сургах"""
        from naive_bayes_model import MyMultinomialNB
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Модель үүсгэх ба сургах
        self.model = MyMultinomialNB(alpha=alpha, ngram_range=ngram_range)
        self.model.fit(self.X_train, self.y_train)
        self.model_type = 'naive_bayes'
        
        return self.model
    
    def train_decision_tree(self, df, attributes, target, max_depth=8, test_size=0.2):
        """Decision Tree сургах"""
        from decision_tree_model import MyDecisionTree
        
        # Train-test split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df[target]
        )
        
        self.X_train = train_df[attributes]
        self.X_test = test_df[attributes]
        self.y_train = train_df[target]
        self.y_test = test_df[target]
        
        # Модель үүсгэх ба сургах
        self.model = MyDecisionTree(max_depth=max_depth)
        self.model.fit(train_df, attributes, target)
        self.model_type = 'decision_tree'
        
        return self.model
    
    def evaluate(self):
        """Моделийг үнэлэх"""
        if self.model is None:
            print("Эхлээд моделийг сургана уу!")
            return None
        
        # Таамаглал хийх
        if self.model_type == 'naive_bayes':
            y_pred = self.model.predict(self.X_test)
        elif self.model_type == 'decision_tree':
            y_pred = self.model.predict(
                pd.DataFrame(self.X_test)
            )
        
        # Accuracy
        acc = accuracy_score(self.y_test, y_pred)
        
        # Confusion matrix
        labels = sorted(set(self.y_test))
        cm = confusion_matrix(self.y_test, y_pred, labels=labels)
        
        # Үр дүн хэвлэх
        print("\n" + "="*60)
        print("МОДЕЛИЙН ҮНЭЛГЭЭ")
        print("="*60)
        print(f"\nАлгоритм: {self.model_type.upper()}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        df_cm = pd.DataFrame(
            cm,
            index=[f"Actual {c}" for c in labels],
            columns=[f"Pred {c}" for c in labels]
        )
        print(df_cm)
        
        # Class-ын дэлгэрэнгүй
        print("\nClass бүрийн мэдээлэл:")
        for i, label in enumerate(labels):
            tp = cm[i][i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n  {label}:")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall:    {recall:.4f}")
            print(f"    F1-Score:  {f1:.4f}")
        
        print("\n" + "="*60)
        
        return {
            'accuracy': acc,
            'confusion_matrix': cm,
            'labels': labels
        }
    
    def predict_comment(self, comment, features=None):
        """Нэг сэтгэгдэл ангилах"""
        if self.model is None:
            return "Эхлээд моделийг сургана уу!"
        
        if self.model_type == 'naive_bayes':
            prediction = self.model.predict_single(comment)
        elif self.model_type == 'decision_tree':
            if features is None:
                return "Decision Tree-д зориулж features оруулна уу!"
            prediction = self.model.predict_single(features)
        
        return prediction
