"""
NLP Service module untuk FAQ System
Implementasi TF-IDF dan similarity matching
"""

import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NLPProcessor:
    def __init__(self, language='indonesian'):
        self.stemmer = PorterStemmer()
        
        # Setup stopwords
        try:
            if language == 'indonesian':
                # Indonesian stopwords
                self.stop_words = set([
                    'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir',
                    'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar',
                    'antara', 'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya',
                    'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai',
                    'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan',
                    'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru',
                    'bawah', 'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah',
                    'begitulah', 'begitupun', 'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah',
                    'benar', 'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya',
                    'berapa', 'berapakah', 'berapalah', 'berapapun', 'berarti', 'berawal', 'berbagai',
                    'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya', 'berjumlah', 'berkali',
                    'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu',
                    'berlangsung', 'berlebihan', 'bermacam', 'bermaksud', 'bermula', 'bersama', 'bersama-sama',
                    'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut', 'berturut-turut',
                    'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa', 'biasanya',
                    'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bukan', 'bukankah',
                    'bukanlah', 'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup', 'cukupkah',
                    'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang',
                    'dekat', 'demi', 'demikian', 'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri',
                    'diakhirinya', 'dialah', 'diantara', 'diantaranya', 'diberi', 'diberikan', 'diberikannya',
                    'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan', 'diibaratkannya',
                    'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya',
                    'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya',
                    'dikira', 'dilakukan', 'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya',
                    'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai', 'dimulailah', 'dimulainya',
                    'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan',
                    'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan',
                    'dipertanyakan', 'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan',
                    'disebutkannya', 'disini', 'disinilah', 'ditambahkan', 'ditandaskan', 'ditanya',
                    'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk', 'ditunjuki',
                    'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan',
                    'diucapkannya', 'diungkapkan', 'dong', 'dua', 'dulu', 'empat', 'enggak', 'enggaknya',
                    'entah', 'entahlah', 'guna', 'gunakan', 'hal', 'hampir', 'hanya', 'hanyalah', 'hari',
                    'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia',
                    'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat',
                    'ingin', 'inginkah', 'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah',
                    'jadi', 'jadilah', 'jadinya', 'jangan', 'jangankan', 'janganlah', 'jauh', 'jawab',
                    'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau',
                    'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah', 'kalaupun',
                    'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun',
                    'karena', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya', 'ke',
                    'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan',
                    'kelihatan', 'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan',
                    'kemungkinannya', 'kenapa', 'kepada', 'kepadanya', 'kesampaian', 'keseluruhan',
                    'keseluruhannya', 'ketika', 'ketujuh', 'kira', 'kira-kira', 'kiranya', 'kita',
                    'kitalah', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu',
                    'lama', 'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam',
                    'maka', 'makanya', 'makin', 'malah', 'malahan', 'mampu', 'mana', 'manakala', 'manalagi',
                    'masa', 'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing',
                    'mau', 'maupun', 'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya',
                    'memang', 'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak',
                    'meminta', 'memintakan', 'memisalkan', 'memperbuat', 'mempergunakan', 'memperkirakan',
                    'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan', 'mempunyai',
                    'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti',
                    'menantikan', 'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan',
                    'mendatang', 'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa',
                    'mengatakan', 'mengatakannya', 'mengenai', 'mengerjakan', 'mengetahui', 'menggunakan',
                    'menghendaki', 'mengibaratkan', 'mengibaratkannya', 'mengingat', 'mengingatkan',
                    'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya', 'mengungkapkan', 'menjadi',
                    'menjawab', 'menjelaskan', 'menuju', 'menunjuk', 'menunjuki', 'menunjukkan',
                    'menunjuknya', 'menurut', 'menuturkan', 'menyampaikan', 'menyangkut', 'menyatakan',
                    'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah', 'merupakan',
                    'meski', 'meskipun', 'meyakini', 'meyakinkan', 'minta', 'mirip', 'misal', 'misalkan',
                    'misalnya', 'mula', 'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'nah',
                    'naik', 'namun', 'nanti', 'nantinya', 'nyaris', 'oleh', 'olehnya', 'pada', 'padanya',
                    'padahal', 'paling', 'panjang', 'pantas', 'para', 'pasti', 'pastilah', 'penting',
                    'pentingnya', 'per', 'percuma', 'perlu', 'perlukah', 'perlunya', 'pernah',
                    'pernahkah', 'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak',
                    'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'ramai', 'rupanya', 'saat', 'saatnya',
                    'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai', 'sampai-sampai',
                    'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah', 'se', 'sebab',
                    'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya',
                    'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum', 'sebelumnya',
                    'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisa', 'sebisanya', 'sebuah',
                    'sebut', 'sebutkan', 'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang',
                    'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya',
                    'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejumlah', 'sekadar',
                    'sekadarnya', 'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun', 'sekarang',
                    'sekarang-sekarang', 'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya',
                    'sekurang-kurangnya', 'sekurangnya', 'sela', 'selain', 'selaku', 'selama', 'selama-lamanya',
                    'selamanya', 'selanjutnya', 'selasa', 'selatan', 'selebar', 'selebihnya', 'selebihnya',
                    'selengkap', 'selengkapnya', 'selentingan', 'selerah', 'selesa', 'selesai', 'selevel',
                    'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa',
                    'semasanya', 'semata', 'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya',
                    'sempat', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah',
                    'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya',
                    'seperti', 'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat',
                    'sesama', 'sesampai', 'sesegera', 'sesekali', 'seseorang', 'sesuai', 'sesuatu',
                    'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah', 'seterusnya',
                    'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi', 'seusai',
                    'sewaktu', 'siap', 'siapa', 'siapakah', 'siapapun', 'sih', 'sini', 'sinilah', 'soal',
                    'soalnya', 'suatu', 'sudah', 'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya',
                    'tahu', 'tahun', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas',
                    'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi', 'tegas', 'tegasnya',
                    'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat',
                    'terakhir', 'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap',
                    'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya',
                    'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan',
                    'tersebut', 'tersebutlah', 'tertentu', 'tertuju', 'terus', 'terutama', 'tetap',
                    'tetapi', 'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah', 'tiga',
                    'tinggi', 'toh', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar',
                    'ujarnya', 'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'waduh',
                    'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong', 'yaitu', 'yakin',
                    'yakni', 'yang'
                ])
            else:
                self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback jika NLTK belum terinstall
            self.stop_words = set(['dan', 'atau', 'adalah', 'dari', 'ke', 'di', 'pada', 'untuk', 'dengan', 'yang'])
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            max_features=5000
        )
        
    def preprocess_text(self, text):
        """Preprocess text untuk cleaning dan normalisasi dengan Indonesian language focus"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Normalize common Indonesian question words
        text = text.replace('bgmn', 'bagaimana')
        text = text.replace('gmn', 'bagaimana') 
        text = text.replace('gimana', 'bagaimana')
        text = text.replace('caranya', 'cara')
        text = text.replace('resetnya', 'reset')
        text = text.replace('passwordnya', 'password')
        text = text.replace('katanya', 'kata')
        text = text.replace('sandinya', 'sandi')
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords dan stemming
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                # Enhanced stemming for Indonesian
                stemmed = token
                
                # Common Indonesian suffixes
                if token.endswith('nya'):
                    stemmed = token[:-3]
                elif token.endswith('kan'):
                    stemmed = token[:-3]
                elif token.endswith('an'):
                    stemmed = token[:-2]
                elif token.endswith('lah'):
                    stemmed = token[:-3]
                elif token.endswith('kah'):
                    stemmed = token[:-3]
                
                # Keep original if stemmed becomes too short
                if len(stemmed) > 2:
                    processed_tokens.append(stemmed)
                else:
                    processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def calculate_similarity(self, query, documents):
        """Calculate cosine similarity antara query dan documents"""
        if not documents:
            return []
        
        # Preprocess query dan documents
        processed_query = self.preprocess_text(query)
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Combine untuk fit vectorizer
        all_texts = [processed_query] + processed_docs
        
        try:
            # Fit dan transform
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            # Fallback ke simple string matching
            return self.simple_string_similarity(processed_query, processed_docs)
    
    def simple_string_similarity(self, query, documents):
        """Simple string similarity sebagai fallback"""
        similarities = []
        query_words = set(query.split())
        
        for doc in documents:
            doc_words = set(doc.split())
            if not query_words or not doc_words:
                similarities.append(0.0)
                continue
                
            intersection = query_words.intersection(doc_words)
            union = query_words.union(doc_words)
            
            similarity = len(intersection) / len(union) if union else 0.0
            similarities.append(similarity)
        
        return similarities
    
    def find_best_match(self, query, faq_data, threshold=0.2):
        """Find best matching FAQ untuk query dengan multiple matching strategies"""
        if not faq_data:
            return None
        
        # Strategy 1: Exact keyword matching
        exact_match = self.find_exact_keyword_match(query, faq_data)
        if exact_match:
            return exact_match
        
        # Strategy 2: TF-IDF similarity
        tfidf_match = self.find_tfidf_match(query, faq_data, threshold)
        if tfidf_match:
            return tfidf_match
        
        # Strategy 3: Simple word overlap dengan threshold rendah
        simple_match = self.find_simple_word_match(query, faq_data, 0.1)
        if simple_match:
            return simple_match
        
        return None
    
    def find_exact_keyword_match(self, query, faq_data):
        """Find exact keyword matches"""
        query_words = set(self.preprocess_text(query).split())
        
        best_match = None
        best_score = 0
        
        for faq in faq_data:
            # Check main question
            question_words = set(self.preprocess_text(faq.get('question', '')).split())
            overlap = len(query_words.intersection(question_words))
            
            if overlap >= 2:  # At least 2 words match
                score = overlap / max(len(query_words), len(question_words))
                if score > best_score:
                    best_score = score
                    best_match = {
                        'faq': faq,
                        'similarity_score': score,
                        'matched_question': faq.get('question', ''),
                        'match_type': 'exact_keyword'
                    }
            
            # Check variations
            variations = faq.get('variations', [])
            for var in variations:
                var_words = set(self.preprocess_text(var.get('variation_question', '')).split())
                overlap = len(query_words.intersection(var_words))
                
                if overlap >= 2:
                    score = overlap / max(len(query_words), len(var_words))
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'faq': faq,
                            'similarity_score': score,
                            'matched_question': var.get('variation_question', ''),
                            'match_type': 'exact_keyword_variation'
                        }
        
        return best_match
    
    def find_tfidf_match(self, query, faq_data, threshold):
        """Find match using TF-IDF similarity"""
        # Extract questions dan answers
        questions = []
        question_map = []
        
        for i, faq in enumerate(faq_data):
            # Include main question
            questions.append(faq.get('question', ''))
            question_map.append({'faq_index': i, 'type': 'main', 'question': faq.get('question', '')})
            
            # Include variations
            variations = faq.get('variations', [])
            for var in variations:
                questions.append(var.get('variation_question', ''))
                question_map.append({
                    'faq_index': i, 
                    'type': 'variation', 
                    'question': var.get('variation_question', '')
                })
        
        # Calculate similarities
        similarities = self.calculate_similarity(query, questions)
        
        if not similarities:
            return None
        
        # Find best match
        best_index = np.argmax(similarities)
        best_score = similarities[best_index]
        
        if best_score < threshold:
            return None
        
        # Map back to FAQ
        match_info = question_map[best_index]
        faq = faq_data[match_info['faq_index']]
        
        return {
            'faq': faq,
            'similarity_score': best_score,
            'matched_question': match_info['question'],
            'match_type': match_info['type']
        }
    
    def find_simple_word_match(self, query, faq_data, threshold):
        """Simple word overlap matching as fallback"""
        query_words = set(self.preprocess_text(query).split())
        
        best_match = None
        best_score = 0
        
        for faq in faq_data:
            # Check main question
            question_words = set(self.preprocess_text(faq.get('question', '')).split())
            if query_words and question_words:
                intersection = query_words.intersection(question_words)
                union = query_words.union(question_words)
                score = len(intersection) / len(union) if union else 0
                
                if score >= threshold and score > best_score:
                    best_score = score
                    best_match = {
                        'faq': faq,
                        'similarity_score': score,
                        'matched_question': faq.get('question', ''),
                        'match_type': 'word_overlap'
                    }
            
            # Check variations
            variations = faq.get('variations', [])
            for var in variations:
                var_words = set(self.preprocess_text(var.get('variation_question', '')).split())
                if query_words and var_words:
                    intersection = query_words.intersection(var_words)
                    union = query_words.union(var_words)
                    score = len(intersection) / len(union) if union else 0
                    
                    if score >= threshold and score > best_score:
                        best_score = score
                        best_match = {
                            'faq': faq,
                            'similarity_score': score,
                            'matched_question': var.get('variation_question', ''),
                            'match_type': 'word_overlap_variation'
                        }
        
        return best_match

# Global NLP processor instance
nlp_processor = NLPProcessor()
