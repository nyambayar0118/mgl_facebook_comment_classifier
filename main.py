"""
main.py - Гол консол application
"""

import os
import sys
from data_loader import DataLoader
from model_evaluator import ModelEvaluator
from visualizations import Visualizer


class SpamClassifierApp:
    """Spam Classifier консол application"""
    
    def __init__(self):
        self.data_loader = None
        self.evaluator = ModelEvaluator()
        self.visualizer = None
        self.data_loaded = False
    
    def clear_screen(self):
        """Дэлгэц цэвэрлэх"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Header хэвлэх"""
        print("\n" + "="*70)
        print(" "*15 + "SPAM/HAM CLASSIFICATION SYSTEM")
        print(" "*20 + "Монгол хэлний сэтгэгдэл ангилагч")
        print("="*70)
    
    def print_menu(self):
        """Үндсэн цэс"""
        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│                         ҮНДСЭН ЦЭС                              │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print("│  1. Өгөгдөл ачаалах                                             │")
        print("│  2. Naive Bayes модель сургах                                   │")
        print("│  3. Decision Tree модель сургах                                 │")
        print("│  4. Сургасан моделийг үнэлэх                                    │")
        print("│  5. Өөрийн сэтгэгдэл ангилуулах                                 │")
        print("│  6. Визуализаци харах                                           │")
        print("│  7. N-gram vocabulary хадгалах (Naive Bayes)                    │")
        print("│  0. Гарах                                                       │")
        print("└─────────────────────────────────────────────────────────────────┘")
    
    def load_data(self):
        """Өгөгдөл ачаалах"""
        self.clear_screen()
        self.print_header()
        print("\n📂 ӨГӨГДӨЛ АЧААЛАХ")
        print("-" * 70)
        
        print("\nЯмар эх үүсвэрээс өгөгдөл ачаалах вэ?")
        print("  1. Google Sheets URL (default)")
        print("  2. CSV файл")
        print("  3. XLSX файл")
        
        choice = input("\nСонголт [1-3]: ").strip()
        
        try:
            if choice == '1' or choice == '':
                print("\n⏳ Google Sheets-ээс өгөгдөл ачаалж байна...")
                self.data_loader = DataLoader(source_type='url')
            elif choice == '2':
                path = input("CSV файлын зам: ").strip()
                self.data_loader = DataLoader(source_type='csv', source_path=path)
            elif choice == '3':
                path = input("XLSX файлын зам: ").strip()
                self.data_loader = DataLoader(source_type='xlsx', source_path=path)
            else:
                print("\n❌ Буруу сонголт!")
                input("\nДарж үргэлжлүүлэх...")
                return
            
            # Өгөгдөл ачаалах
            df = self.data_loader.load_data()
            self.visualizer = Visualizer(df)
            self.data_loaded = True
            
            print(f"\n✅ Амжилттай ачаалагдлаа!")
            print(f"   Нийт мөр: {len(df)}")
            print(f"   Label тархалт:")
            print(df['label'].value_counts())
            
        except Exception as e:
            print(f"\n❌ Алдаа гарлаа: {e}")
        
        input("\nДарж үргэлжлүүлэх...")
    
    def train_naive_bayes(self):
        """Naive Bayes сургах"""
        if not self.data_loaded:
            print("\n❌ Эхлээд өгөгдөл ачаална уу! (Сонголт 1)")
            input("\nДарж үргэлжлүүлэх...")
            return
        
        self.clear_screen()
        self.print_header()
        print("\n🤖 NAIVE BAYES МОДЕЛЬ СУРГАХ")
        print("-" * 70)
        
        # Текстийн эх үүсвэр сонгох
        print("\nТекстийн эх үүсвэр сонгох:")
        print("  1. Raw comment (Анхны сэтгэгдэл)")
        print("  2. Transliterated comment (Цэвэрлэсэн сэтгэгдэл)")
        print("  3. Both (Хоёулаа нэгтгэсэн)")
        
        text_choice = input("\nСонголт [1-3] [1]: ").strip()
        
        if text_choice == '2':
            text_source = 'transliterated'
            print("✓ Цэвэрлэсэн сэтгэгдэл ашиглана")
        elif text_choice == '3':
            text_source = 'both'
            print("✓ Хоёр баганыг нэгтгэж ашиглана")
        else:
            text_source = 'raw'
            print("✓ Анхны сэтгэгдэл ашиглана")
        
        # Параметрүүд асуух
        print("\nПараметрүүд:")
        alpha = input("  Laplace smoothing α [1.0]: ").strip()
        alpha = float(alpha) if alpha else 1.0
        
        ngram_type = input("  N-gram төрөл (1=unigram, 2=unigram+bigram, 3=unigram+bigram+trigram) [2]: ").strip()
        ngram_type = int(ngram_type) if ngram_type else 2
        ngram_range = (1, ngram_type)
        
        test_size = input("  Test хэсгийн хувь (0.2 = 20%) [0.3]: ").strip()
        test_size = float(test_size) if test_size else 0.3
        
        try:
            print("\n⏳ Моделийг сургаж байна...")
            X, y = self.data_loader.prepare_for_naive_bayes(text_source=text_source)
            self.evaluator.train_naive_bayes(X, y, alpha=alpha, ngram_range=ngram_range, test_size=test_size)
            
            print("\n✅ Модель амжилттай сургагдлаа!")
            print(f"   Текстийн эх үүсвэр: {text_source}")
            print(f"   Vocabulary хэмжээ: {len(self.evaluator.model.vocabulary_)}")
            print(f"   Train set: {len(self.evaluator.X_train)} мөр")
            print(f"   Test set: {len(self.evaluator.X_test)} мөр")
            
            # Vocabulary хадгалах эсэхийг асуух
            save_vocab = input("\nN-gram vocabulary-г файл руу хадгалах уу? (y/n) [y]: ").strip().lower()
            if save_vocab != 'n':
                filename = input("Файлын нэр [vocabulary.txt]: ").strip()
                if not filename:
                    filename = 'vocabulary.txt'
                self.evaluator.model.save_vocabulary(filename)
            
        except Exception as e:
            print(f"\n❌ Алдаа гарлаа: {e}")
        
        input("\nДарж үргэлжлүүлэх...")
    
    def train_decision_tree(self):
        """Decision Tree сургах"""
        if not self.data_loaded:
            print("\n❌ Эхлээд өгөгдөл ачаална уу! (Сонголт 1)")
            input("\nДарж үргэлжлүүлэх...")
            return
        
        self.clear_screen()
        self.print_header()
        print("\n🌳 DECISION TREE МОДЕЛЬ СУРГАХ")
        print("-" * 70)
        
        # Параметрүүд асуух
        print("\nПараметрүүд:")
        max_depth = input("  Maximum depth [8]: ").strip()
        max_depth = int(max_depth) if max_depth else 8
        
        test_size = input("  Test хэсгийн хувь (0.2 = 20%) [0.2]: ").strip()
        test_size = float(test_size) if test_size else 0.2
        
        try:
            print("\n⏳ Моделийг сургаж байна...")
            df, attributes, target = self.data_loader.prepare_for_decision_tree()
            self.evaluator.train_decision_tree(df, attributes, target, max_depth=max_depth, test_size=test_size)
            
            print("\n✅ Модель амжилттай сургагдлаа!")
            print(f"   Train set: {len(self.evaluator.X_train)} мөр")
            print(f"   Test set: {len(self.evaluator.X_test)} мөр")
            
            # Tree бүтцийг харуулах эсэхийг асуух
            show_tree = input("\nTree бүтцийг харах уу? (y/n) [n]: ").strip().lower()
            if show_tree == 'y':
                print("\n--- Decision Tree бүтэц ---")
                self.evaluator.model.print_tree()
            
        except Exception as e:
            print(f"\n❌ Алдаа гарлаа: {e}")
        
        input("\nДарж үргэлжлүүлэх...")
    
    def evaluate_model(self):
        """Моделийг үнэлэх"""
        if self.evaluator.model is None:
            print("\n❌ Эхлээд моделийг сургана уу! (Сонголт 2 эсвэл 3)")
            input("\nДарж үргэлжлүүлэх...")
            return
        
        self.clear_screen()
        self.print_header()
        print("\n📊 МОДЕЛИЙН ҮНЭЛГЭЭ")
        print("-" * 70)
        
        try:
            self.evaluator.evaluate()
        except Exception as e:
            print(f"\n❌ Алдаа гарлаа: {e}")
        
        input("\nДарж үргэлжлүүлэх...")
    
    def classify_comment(self):
        """Өөрийн сэтгэгдэл ангилуулах"""
        if self.evaluator.model is None:
            print("\n❌ Эхлээд моделийг сургана уу! (Сонголт 2 эсвэл 3)")
            input("\nДарж үргэлжлүүлэх...")
            return
        
        self.clear_screen()
        self.print_header()
        print("\n💬 СЭТГЭГДЭЛ АНГИЛУУЛАХ")
        print("-" * 70)
        print(f"\nОдоогийн модель: {self.evaluator.model_type.upper()}")
        
        if self.evaluator.model_type == 'naive_bayes':
            self._classify_with_naive_bayes()
        elif self.evaluator.model_type == 'decision_tree':
            self._classify_with_decision_tree()
    
    def _classify_with_naive_bayes(self):
        """Naive Bayes-аар ангилах"""
        while True:
            print("\n" + "-" * 70)
            comment = input("\nСэтгэгдэл оруулна уу (буцах бол 'q'): ").strip()
            
            if comment.lower() == 'q':
                break
            
            if not comment:
                print("⚠️  Сэтгэгдэл хоосон байна!")
                continue
            
            try:
                prediction = self.evaluator.predict_comment(comment)
                
                print("\n" + "="*70)
                if prediction == 'spam':
                    print("🚫 Таамаглал: SPAM")
                else:
                    print("✅ Таамаглал: HAM (Normal comment)")
                print("="*70)
                
            except Exception as e:
                print(f"\n❌ Алдаа гарлаа: {e}")
        
        input("\nДарж үргэлжлүүлэх...")
    
    def _classify_with_decision_tree(self):
        """Decision Tree-ээр ангилах"""
        print("\n⚠️  Decision Tree нь шинжүүдийг шаарддаг.")
        print("Дараах шинжүүдийг оруулна уу:")
        
        features_names = [
            "Зураг агуулсан эсэх",
            "Нэрээ нууцалсан эсэх",
            "Монгол нэр эсэх",
            "Кирил, латин биш тэмдэгт ашигласан эсэх",
            "Email агуулсан эсэх",
            "Link агуулсан эсэх",
            "Утасны дугаар агуулсан эсэх",
            "Зөв бичсэн хувь",
            "Ашигласан үсэг",
            "Emoji-ний тоо",
            "Сэтгэгдлийн урт"
        ]
        
        while True:
            print("\n" + "-" * 70)
            cont = input("\nШинж оруулах уу? (y/n) [y]: ").strip().lower()
            if cont == 'n':
                break
            
            features = {}
            print()
            for fname in features_names:
                val = input(f"  {fname}: ").strip()
                try:
                    features[fname] = float(val) if val else 0
                except:
                    features[fname] = 0
            
            try:
                prediction = self.evaluator.predict_comment(None, features=features)
                
                print("\n" + "="*70)
                if prediction == 'spam':
                    print("🚫 Таамаглал: SPAM")
                else:
                    print("✅ Таамаглал: HAM (Normal comment)")
                print("="*70)
                
            except Exception as e:
                print(f"\n❌ Алдаа гарлаа: {e}")
        
        input("\nДарж үргэлжлүүлэх...")
    
    def show_visualizations(self):
        """Визуализаци харуулах"""
        if not self.data_loaded:
            print("\n❌ Эхлээд өгөгдөл ачаална уу! (Сонголт 1)")
            input("\nДарж үргэлжлүүлэх...")
            return
        
        while True:
            self.clear_screen()
            self.print_header()
            print("\n📊 ВИЗУАЛИЗАЦИ")
            print("-" * 70)
            print("\n  1. Сэтгэгдлийн урт тархалт")
            print("  2. Emoji тоо тархалт")
            print("  3. Үсгийн төрөл (Pie chart)")
            print("  4. Binary features хувь тархалт")
            print("  5. Feature=1 үед spam/ham тархалт")
            print("  6. Correlation matrix")
            print("  7. Бүгдийг харуулах")
            print("  0. Буцах")
            
            choice = input("\nСонголт [0-7]: ").strip()
            
            try:
                if choice == '1':
                    self.visualizer.plot_comment_length()
                elif choice == '2':
                    self.visualizer.plot_emoji_count()
                elif choice == '3':
                    self.visualizer.plot_script_types()
                elif choice == '4':
                    self.visualizer.plot_binary_distribution()
                elif choice == '5':
                    self.visualizer.plot_spam_ham_by_feature()
                elif choice == '6':
                    self.visualizer.plot_correlation_matrix()
                elif choice == '7':
                    print("\n⏳ Бүх графикийг үүсгэж байна...")
                    self.visualizer.plot_comment_length()
                    self.visualizer.plot_emoji_count()
                    self.visualizer.plot_script_types()
                    self.visualizer.plot_binary_distribution()
                    self.visualizer.plot_spam_ham_by_feature()
                    self.visualizer.plot_correlation_matrix()
                elif choice == '0':
                    break
                else:
                    print("\n❌ Буруу сонголт!")
                    input("\nДарж үргэлжлүүлэх...")
            except Exception as e:
                print(f"\n❌ Алдаа гарлаа: {e}")
                input("\nДарж үргэлжлүүлэх...")
    
    def save_vocabulary_to_file(self):
        """N-gram vocabulary файл руу хадгалах"""
        if self.evaluator.model is None or self.evaluator.model_type != 'naive_bayes':
            print("\n❌ Эхлээд Naive Bayes моделийг сургана уу! (Сонголт 2)")
            input("\nДарж үргэлжлүүлэх...")
            return
        
        self.clear_screen()
        self.print_header()
        print("\n📝 N-GRAM VOCABULARY ХАДГАЛАХ")
        print("-" * 70)
        
        filename = input("\nФайлын нэр [vocabulary.txt]: ").strip()
        if not filename:
            filename = 'vocabulary.txt'
        
        try:
            self.evaluator.model.save_vocabulary(filename)
            print(f"\n✅ Амжилттай хадгалагдлаа: {filename}")
            print(f"   Нийт n-grams: {len(self.evaluator.model.vocabulary_)}")
        except Exception as e:
            print(f"\n❌ Алдаа гарлаа: {e}")
        
        input("\nДарж үргэлжлүүлэх...")
    
    def run(self):
        """Application ажиллуулах"""
        while True:
            self.clear_screen()
            self.print_header()
            
            # Статус харуулах
            print("\n📌 Статус:")
            print(f"   Өгөгдөл: {'✅ Ачаалагдсан' if self.data_loaded else '❌ Ачаалаагүй'}")
            print(f"   Модель: {'✅ ' + self.evaluator.model_type.upper() if self.evaluator.model else '❌ Сургаагүй'}")
            
            self.print_menu()
            
            choice = input("\nСонголт [0-7]: ").strip()
            
            if choice == '1':
                self.load_data()
            elif choice == '2':
                self.train_naive_bayes()
            elif choice == '3':
                self.train_decision_tree()
            elif choice == '4':
                self.evaluate_model()
            elif choice == '5':
                self.classify_comment()
            elif choice == '6':
                self.show_visualizations()
            elif choice == '7':
                self.save_vocabulary_to_file()
            elif choice == '0':
                print("\n👋 Баяртай!")
                break
            else:
                print("\n❌ Буруу сонголт!")
                input("\nДарж үргэлжлүүлэх...")


if __name__ == "__main__":
    app = SpamClassifierApp()
    app.run()
