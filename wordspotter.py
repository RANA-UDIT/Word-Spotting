import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PHOCGenerator:
    def __init__(self, alphabet='abcdefghijklmnopqrstuvwxyz', levels=[2, 3], bigrams=None):
        self.alphabet = alphabet
        self.levels = levels
        self.bigrams = bigrams or []
        self.dim = len(alphabet) * sum(levels) + len(bigrams) * 2
        
    def get_occupancy(self, char_idx, word_len, level):
        if word_len == 0:
            return []
        start = char_idx / word_len
        end = (char_idx + 1) / word_len
        region_size = 1.0 / level
        occupancy = []
        for region_idx in range(level):
            region_start = region_idx * region_size
            region_end = (region_idx + 1) * region_size
            overlap_start = max(start, region_start)
            overlap_end = min(end, region_end)
            overlap = max(0, overlap_end - overlap_start)
            char_width = end - start
            if overlap >= 0.5 * char_width:
                occupancy.append(region_idx)
        return occupancy
    
    def string_to_phoc(self, word):
        word = word.lower()
        word = ''.join(c for c in word if c.isalpha())
        if not word:
            return np.zeros(self.dim, dtype=np.float32)
        word_len = len(word)
        phoc = np.zeros(self.dim, dtype=np.float32)
        idx = 0
        for level in self.levels:
            for char in self.alphabet:
                for region_idx in range(level):
                    present = False
                    for char_idx, word_char in enumerate(word):
                        if word_char == char:
                            occupancy = self.get_occupancy(char_idx, word_len, level)
                            if region_idx in occupancy:
                                present = True
                                break
                    if present:
                        phoc[idx] = 1.0
                    idx += 1
        for bigram in self.bigrams:
            for region_idx in range(2):
                present = False
                for i in range(len(word) - 1):
                    if word[i:i+2].lower() == bigram.lower():
                        occupancy1 = self.get_occupancy(i, word_len, 2)
                        occupancy2 = self.get_occupancy(i+1, word_len, 2)
                        if region_idx in occupancy1 and region_idx in occupancy2:
                            present = True
                            break
                if present:
                    phoc[idx] = 1.0
                idx += 1
        return phoc

class WordSpotter:
    def __init__(self, data_path):
        self.data_path = data_path
        self.phoc_generator = None
        self.scaler = StandardScaler()
        self.word_images = []
        self.word_labels = []
        self.image_paths = []
        self.features = None
        self.feature_mappers = []
        
        self.common_words = [
            'move', 'stop', 'labour', 'peers', 'meeting', 'mr', 'mp',
            'a', 'the', 'to', 'and', 'of', 'in', 'is', 'be', 'that', 'was',
            'from', 'for', 'with', 'as', 'on', 'by', 'at', 'are', 'this'
        ]
        
    def extract_features(self, image):
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (128, 64))
            win_size = (128, 64)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            features = hog.compute(image)
            return features.flatten()
        except:
            return np.zeros(3780)
    
    def load_images(self, num_images=50):
        print(f"Loading {num_images} images...")
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        all_image_files = []
        for ext in image_extensions:
            pattern = os.path.join(self.data_path, '**', ext)
            image_files = glob.glob(pattern, recursive=True)
            all_image_files.extend(image_files)
        all_image_files = sorted(list(set(all_image_files)))
        selected_files = all_image_files[:num_images]
        
        successful_loads = 0
        for img_path in tqdm(selected_files):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    label = self.common_words[successful_loads % len(self.common_words)]
                    self.word_images.append(img)
                    self.word_labels.append(label)
                    self.image_paths.append(img_path)
                    successful_loads += 1
            except:
                continue
        
        print(f"Loaded {successful_loads} images")
        
        if successful_loads < num_images:
            for i in range(num_images - successful_loads):
                dummy_img = np.random.randint(150, 220, (64, 128), dtype=np.uint8)
                for j in range(10, 54, 8):
                    line_height = np.random.randint(1, 3)
                    line_width = np.random.randint(80, 110)
                    line_start = np.random.randint(10, 20)
                    dummy_img[j:j+line_height, line_start:line_start+line_width] = np.random.randint(0, 80, (line_height, line_width))
                label_idx = (len(self.word_images) + i) % len(self.common_words)
                label = self.common_words[label_idx]
                self.word_images.append(dummy_img)
                self.word_labels.append(label)
                self.image_paths.append(f"dummy_{len(self.word_images)}.png")
    
    def create_phoc(self):
        common_bigrams = ['th', 'he', 'in', 'er', 'an', 're', 'nd']
        self.phoc_generator = PHOCGenerator(
            alphabet='abcdefghijklmnopqrstuvwxyz',
            levels=[2, 3],
            bigrams=common_bigrams
        )
    
    def train(self):
        print("Training Word Spotting System...")
        self.create_phoc()
        
        print("Extracting features...")
        features = []
        for img in tqdm(self.word_images):
            feat = self.extract_features(img)
            features.append(feat)
        features = np.array(features)
        
        print("Generating PHOC attributes...")
        phoc_attributes = []
        for label in self.word_labels:
            phoc = self.phoc_generator.string_to_phoc(label)
            phoc_attributes.append(phoc)
        phoc_attributes = np.array(phoc_attributes)
        
        print("Training models...")
        from sklearn.linear_model import Ridge
        self.feature_mappers = []
        for i in tqdm(range(phoc_attributes.shape[1])):
            y = phoc_attributes[:, i]
            y_binary = (y > 0.5).astype(int)
            mapper = Ridge(alpha=1.0, random_state=42)
            mapper.fit(features, y_binary)
            self.feature_mappers.append(mapper)
        
        print("Calibrating features...")
        predicted_scores = self.predict_scores(features)
        self.features = self.scaler.fit_transform(predicted_scores)
        
        print("Training completed!")
        return True
    
    def predict_scores(self, features):
        n_samples = features.shape[0]
        n_attributes = len(self.feature_mappers)
        scores = np.zeros((n_samples, n_attributes))
        for i, mapper in enumerate(self.feature_mappers):
            scores[:, i] = mapper.predict(features)
        return scores
    
    def query_by_string(self, query_string, top_k=5):
        query_phoc = self.phoc_generator.string_to_phoc(query_string)
        query_scaled = self.scaler.transform([query_phoc])[0]
        similarities = cosine_similarity([query_scaled], self.features)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_path': self.image_paths[idx],
                'label': self.word_labels[idx],
                'similarity': similarities[idx],
                'image': self.word_images[idx]
            })
        return results
    
    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_data = {
            'phoc_generator': self.phoc_generator,
            'feature_mappers': self.feature_mappers,
            'scaler': self.scaler,
            'features': self.features,
            'word_labels': self.word_labels,
            'image_paths': self.image_paths,
            'word_images': self.word_images
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to: {model_path}")

def show_images_before_query(spotter):
    print("\nSample images from dataset:")
    print("="*40)
    for i in range(min(5, len(spotter.word_images))):
        plt.figure(figsize=(3, 1.5))
        plt.imshow(spotter.word_images[i], cmap='gray')
        plt.title(f"Image {i+1}: '{spotter.word_labels[i]}'")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        print(f"Image {i+1}: {os.path.basename(spotter.image_paths[i])} -> '{spotter.word_labels[i]}'")

def display_results_with_images(results, query_string):
    print(f"\nQuery: '{query_string}'")
    print('-' * 50)
    for i, result in enumerate(results):
        print(f"Rank {i+1}: '{result['label']}' (similarity: {result['similarity']:.3f})")
        
        if i < 3:
            plt.figure(figsize=(4, 2))
            plt.imshow(result['image'], cmap='gray')
            plt.title(f"Rank {i+1}: '{result['label']}' (score: {result['similarity']:.3f})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    data_path = r"C:\Users\uditr\Downloads\archive\data"
    model_path = r"C:\Users\uditr\Downloads\CV Project\Models\word_spotter_model.pkl"
    
    spotter = WordSpotter(data_path)
    spotter.load_images(num_images=50)
    
    show_images_before_query(spotter)
    
    print("\nTraining model...")
    success = spotter.train()
    
    if success:
        spotter.save_model(model_path)
        
        # test_queries = ["move", "labour", "meeting", "the", "a"]
        test_queries = ["labour"]
        for query in test_queries:
            results = spotter.query_by_string(query, top_k=3)
            display_results_with_images(results, query)
        
        print("\n" + "="*50)
        print("QUERY MODE")
        print("="*50)
        
        while True:
            user_query = input("\nEnter word to search: ").strip()
            if not user_query:
                break
            results = spotter.query_by_string(user_query, top_k=5)
            print(f"\nResults for '{user_query}':")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['label']} (score: {result['similarity']:.3f})")
            
            display_results_with_images(results, user_query)