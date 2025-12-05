"""
data_loader.py - Өгөгдөл ачааллах болон боловсруулах модуль
"""

import pandas as pd
import ssl

# HTTPS сертификатын алдааг алгасах
ssl._create_default_https_context = ssl._create_unverified_context


class DataLoader:
    """Өгөгдөл ачааллах класс"""
    
    def __init__(self, source_type='url', source_path=None):
        """
        source_type: 'url', 'csv', эсвэл 'xlsx'
        source_path: Файлын зам эсвэл URL
        """
        self.source_type = source_type
        self.source_path = source_path
        self.df_raw = None
        
    def load_data(self):
        """Өгөгдөл ачаалах"""
        if self.source_type == 'url':
            url = self.source_path or 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQC0C_5Sim0DhMAZbxRNxpFxBajpiHEGKeGWtSLLgbUWizbdrtoNGvv-x0P1phRr7DT6J5rCVyGzaYF/pub?output=csv'
            df = pd.read_csv(url, engine='python', on_bad_lines='skip')
        elif self.source_type == 'csv':
            df = pd.read_csv(self.source_path, on_bad_lines='skip')
        elif self.source_type == 'xlsx':
            df = pd.read_excel(self.source_path)
        else:
            raise ValueError(f"Танихгүй source_type: {self.source_type}")
        
        # df_raw-д хадгалах
        self.df_raw = df.copy()
        
        # Эхний мөрийг header болгох (хэрэв шаардлагатай бол)
        if self.source_type in ['url', 'csv']:
            self.df_raw.columns = self.df_raw.iloc[0].astype(str).str.strip()
            self.df_raw = self.df_raw.iloc[1:]
        
        # label цэвэрлэх
        if 'label' in self.df_raw.columns:
            self.df_raw['label'] = self.df_raw['label'].str.strip().str.lower()
        
        return self.df_raw
    
    def prepare_for_naive_bayes(self, text_source='raw'):
        """
        Naive Bayes-д зориулж өгөгдөл бэлдэх
        
        text_source: 'raw', 'transliterated', эсвэл 'both'
        """
        if self.df_raw is None:
            self.load_data()
        
        df_clf = self.df_raw[['label']].copy()
        
        if text_source == 'raw':
            # Зөвхөн Raw comment
            df_clf['text'] = self.df_raw['Raw comment']
        elif text_source == 'transliterated':
            # Зөвхөн Цэвэрлэсэн сэтгэгдэл
            if 'Цэвэрлэсэн сэтгэгдэл' in self.df_raw.columns:
                df_clf['text'] = self.df_raw['Цэвэрлэсэн сэтгэгдэл']
            else:
                print("⚠️  'Цэвэрлэсэн сэтгэгдэл' багана олдсонгүй, Raw comment ашиглана.")
                df_clf['text'] = self.df_raw['Raw comment']
        elif text_source == 'both':
            # Хоёулаа нэгтгэх
            raw = self.df_raw['Raw comment'].fillna('')
            if 'Цэвэрлэсэн сэтгэгдэл' in self.df_raw.columns:
                trans = self.df_raw['Цэвэрлэсэн сэтгэгдэл'].fillna('')
                df_clf['text'] = raw + ' ' + trans
            else:
                print("⚠️  'Цэвэрлэсэн сэтгэгдэл' багана олдсонгүй, Raw comment ашиглана.")
                df_clf['text'] = raw
        else:
            raise ValueError(f"Танихгүй text_source: {text_source}")
        
        df_clf = df_clf.dropna(subset=['text', 'label'])
        
        return df_clf['text'], df_clf['label']
    
    def prepare_for_decision_tree(self):
        """Decision Tree-д зориулж өгөгдөл бэлдэх"""
        if self.df_raw is None:
            self.load_data()
        
        df = self.df_raw.copy()
        
        # Label багана
        COL_LABEL = "label"
        df = df.dropna(subset=[COL_LABEL])
        
        # Boolean багануудыг боловсруулах
        BOOL_COLUMNS = [
            "Зураг агуулсан эсэх",
            "Нэрээ нууцалсан эсэх",
            "Монгол нэр эсэх",
            "Кирил, латин биш тэмдэгт ашигласан эсэх",
            "Email агуулсан эсэх",
            "Link агуулсан эсэх",
            "Утасны дугаар агуулсан эсэх"
        ]
        
        # Тоон багануудыг боловсруулах
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
                    "Yes": 1, "No": 0,
                    "тийм": 1, "үгүй": 0
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
        
        attributes = BOOL_COLUMNS + NUMERIC_COLUMNS
        
        return df, attributes, COL_LABEL
    
    def get_visualization_data(self):
        """Визуализацид зориулж өгөгдөл бэлдэх"""
        if self.df_raw is None:
            self.load_data()
        
        return self.df_raw.copy()
