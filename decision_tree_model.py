"""
decision_tree_model.py - Decision Tree модель
"""

import pandas as pd
import numpy as np


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


def print_tree(tree, indent=""):
    """Decision Tree-г хүснэгт маягаар хэвлэх"""
    if not isinstance(tree, dict):
        print(indent + "→ " + str(tree))
        return
    
    for attr, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}[{attr} = {value}]")
            print_tree(subtree, indent + "   ")


class MyDecisionTree:
    """Decision Tree wrapper класс"""
    
    def __init__(self, max_depth=8):
        self.max_depth = max_depth
        self.tree = None
        self.attributes = None
        self.target = None
    
    def fit(self, df, attributes, target):
        """Decision Tree сургах"""
        self.attributes = attributes
        self.target = target
        self.tree = decision_tree_learning(
            df[attributes + [target]],
            attributes,
            target,
            max_depth=self.max_depth
        )
    
    def predict(self, df):
        """Таамаглал хийх"""
        return predict_dataset(self.tree, df[self.attributes])
    
    def predict_single(self, sample_dict):
        """Нэг дээжинд таамаглал хийх"""
        # dict -> pandas Series
        sample = pd.Series(sample_dict)
        return predict(self.tree, sample)
    
    def print_tree(self):
        """Tree хэвлэх"""
        print_tree(self.tree)
