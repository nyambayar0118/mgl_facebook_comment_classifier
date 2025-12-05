"""
visualizations.py - Визуализацийн модуль
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Visualizer:
    """Визуализаци үүсгэх класс"""
    
    def __init__(self, df_raw):
        self.df_raw = df_raw.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Өгөгдөл бэлдэх"""
        # Тоон утгуудыг боловсруулах
        self.df_raw['Сэтгэгдлийн урт'] = pd.to_numeric(
            self.df_raw['Сэтгэгдлийн урт'], errors='coerce'
        )
        self.df_raw['Emoji-ний тоо'] = pd.to_numeric(
            self.df_raw['Emoji-ний тоо'], errors='coerce'
        ).fillna(0).astype(int)
        
        # Binary features боловсруулах
        rename_map = {
            'Зураг агуулсан эсэх': 'contains_photo',
            'Нэрээ нууцалсан эсэх': 'is_anonymous',
            'Монгол нэр эсэх': 'is_mongolian_name',
            'Кирил, латин биш тэмдэгт ашигласан эсэх': 'has_weird_chars',
            'Email агуулсан эсэх': 'contains_email',
            'Link агуулсан эсэх': 'contains_link',
            'Утасны дугаар агуулсан эсэх': 'contains_number'
        }
        self.df_raw = self.df_raw.rename(columns=rename_map)
        
        self.binary_features = [
            'contains_photo', 'is_anonymous', 'is_mongolian_name',
            'has_weird_chars', 'contains_email', 'contains_link', 'contains_number'
        ]
        
        # Binary features-г 0/1 болгох
        for col in self.binary_features:
            if col in self.df_raw.columns:
                self.df_raw[col] = (
                    self.df_raw[col].astype(str).str.strip().str.lower()
                    .replace({'yes':1, 'no':0, 'true':1, 'false':0,
                              'тийм':1, 'үгүй':0, '1':1, '0':0})
                )
                self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce').fillna(0).astype(int)
    
    def plot_comment_length(self):
        """Сэтгэгдлийн урт histogram"""
        df_len = self.df_raw[['label', 'Сэтгэгдлийн урт']].dropna()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=df_len,
            x='Сэтгэгдлийн урт',
            hue='label',
            bins=10,
            multiple='dodge',
            stat='count',
            palette={'spam': '#ffcccc', 'ham': '#ccffcc'}
        )
        plt.title("Comment length distribution by class (spam vs ham)")
        plt.xlabel("Сэтгэгдлийн урт (тэмдэгтүүдийн тоо)")
        plt.ylabel("Давтамж")
        
        max_len = int(df_len['Сэтгэгдлийн урт'].max())
        step = 200
        plt.xticks(range(0, max_len + step, step), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_emoji_count(self):
        """Emoji тоо histogram"""
        df_plot = self.df_raw.copy()
        df_plot['Emoji-ний тоо'] = df_plot['Emoji-ний тоо'].clip(upper=10)
        
        plt.figure(figsize=(12, 5))
        sns.histplot(
            data=df_plot,
            x='Emoji-ний тоо',
            hue='label',
            bins=10,
            multiple='dodge',
            stat='count',
            palette={'spam': '#ffcccc', 'ham': '#ccffcc'}
        )
        
        plt.title("Emoji count distribution by class (spam vs ham)")
        plt.xlabel("Emoji-ний тоо (≥10 → 10 гэж нэгтгэсэн)")
        plt.ylabel("Давтамж")
        plt.xticks(range(0, 11))
        plt.tight_layout()
        plt.show()
    
    def plot_script_types(self):
        """Үсгийн төрөл pie chart"""
        plt.figure(figsize=(10, 5))
        
        labels = ['spam', 'ham']
        colors_spam = ['#ff9999', '#ffcccc', '#ff6666', '#ffb3b3']
        colors_ham = ['#99ff99', '#ccffcc', '#66ff66', '#b3ffb3']
        colors_map = {'spam': colors_spam, 'ham': colors_ham}
        
        for i, lbl in enumerate(labels):
            plt.subplot(1, 2, i+1)
            subset = self.df_raw[self.df_raw['label'] == lbl]
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
    
    def plot_binary_distribution(self):
        """Binary features хувь тархалт"""
        chunk_size = 3
        total_rows = len(self.df_raw)
        
        for start in range(0, len(self.binary_features), chunk_size):
            end = start + chunk_size
            chunk = self.binary_features[start:end]
            
            plt.figure(figsize=(18, 6))
            
            for i, col in enumerate(chunk):
                plt.subplot(1, len(chunk), i+1)
                
                counts = self.df_raw[col].value_counts().sort_index()
                pct = (counts / total_rows * 100).round(2)
                
                ax = sns.barplot(
                    x=counts.index.astype(str),
                    y=pct.values,
                    palette=["#ffcccc", "#ccffcc"]
                )
                
                plt.title(f"{col} — % Distribution", fontsize=12)
                plt.xlabel("0 = No, 1 = Yes")
                plt.ylabel("Percentage (%)")
                plt.ylim(0, 100)
                
                for index, value in enumerate(pct.values):
                    count_value = counts.iloc[index]
                    ax.text(
                        index, value + 2,
                        f"{value}%\n({count_value})",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold'
                    )
            
            plt.tight_layout()
            plt.show()
    
    def plot_spam_ham_by_feature(self):
        """Feature=1 үед spam/ham тархалт"""
        chunk_size = 3
        
        for start in range(0, len(self.binary_features), chunk_size):
            end = start + chunk_size
            chunk = self.binary_features[start:end]
            
            plt.figure(figsize=(18, 6))
            
            for i, col in enumerate(chunk):
                plt.subplot(1, len(chunk), i+1)
                
                df_feature = self.df_raw[self.df_raw[col] == 1]
                total_with_feature = len(df_feature)
                
                if total_with_feature == 0:
                    plt.title(f"{col} — No rows with value 1")
                    plt.bar(["ham", "spam"], [0, 0])
                    continue
                
                spam_count = (df_feature["label"] == "spam").sum()
                ham_count = (df_feature["label"] == "ham").sum()
                
                spam_pct = round(spam_count / total_with_feature * 100, 2)
                ham_pct = round(ham_count / total_with_feature * 100, 2)
                
                values = [ham_pct, spam_pct]
                labels = ["ham", "spam"]
                
                ax = sns.barplot(
                    x=labels, y=values,
                    palette={'spam': '#ffcccc', 'ham': '#ccffcc'}
                )
                
                plt.title(f"{col} — Among feature=1", fontsize=12)
                plt.ylabel("% of comments (feature=1)")
                plt.ylim(0, 100)
                
                for idx, val in enumerate(values):
                    count_val = ham_count if idx == 0 else spam_count
                    ax.text(
                        idx, val + 2,
                        f"{val}%\n({count_val} comments)",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold'
                    )
            
            plt.tight_layout()
            plt.show()
    
    def plot_correlation_matrix(self):
        """Correlation matrix"""
        heatmap_cols = ['Сэтгэгдлийн урт', 'Emoji-ний тоо'] + self.binary_features
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            self.df_raw[heatmap_cols].corr(),
            annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5
        )
        plt.title("Correlation Matrix for All Numerical & Binary Features")
        plt.tight_layout()
        plt.show()
