import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PHOC:
    def __init__(self, levels=[2], bigrams=5):
        self.levels = levels
        self.bigrams = bigrams
        self.most_frequent_bigrams = [
            'th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd'
        ][:bigrams]
    
    def __call__(self, word):
        word = word.lower().strip()
        n = len(word)
        if n == 0:
            return np.zeros(self.compute_dimension())
        
        result = []
        
        for level in self.levels:
            for region in range(level):
                start = region / level
                end = (region + 1) / level
                
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    char_found = False
                    for i, word_char in enumerate(word):
                        char_pos = (i + 0.5) / n
                        if word_char == char and start <= char_pos < end:
                            result.append(1)
                            char_found = True
                            break
                    if not char_found:
                        result.append(0)
        
        for i in range(2):
            if i == 0:
                text_part = word[:max(1, len(word)//2)]
            else:
                text_part = word[len(word)//2:]
            
            for bigram in self.most_frequent_bigrams:
                result.append(1 if bigram in text_part else 0)
        
        return np.array(result, dtype=np.float32)
    
    def compute_dimension(self):
        return sum(self.levels) * 26 + 2 * self.bigrams

class RobustTorchPHOCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RobustTorchPHOCModel, self).__init__()
        # Match the architecture from training (simple model)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),  # Small model as in training
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class FisherVector:
    def __init__(self, vocab_size=4):
        self.vocab_size = vocab_size
        self.gmm = None
        self.pca = None
        
    def transform(self, descriptors):
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(2 * 32 * self.vocab_size)  # 32 from PCA
        
        descriptors_reduced = self.pca.transform(descriptors)
        
        means = self.gmm.means_
        covs = self.gmm.covariances_
        weights = self.gmm.weights_
        posterior = self.gmm.predict_proba(descriptors_reduced)
        
        grad_means = np.zeros_like(means)
        grad_vars = np.zeros_like(covs)
        
        for k in range(self.vocab_size):
            diff = descriptors_reduced - means[k]
            grad_means[k] = np.sum(posterior[:, k:k+1] * diff, axis=0) / (np.sqrt(weights[k]) * np.sqrt(covs[k]))
            
            diff_sq = (descriptors_reduced - means[k]) ** 2
            grad_vars[k] = np.sum(posterior[:, k:k+1] * (diff_sq / covs[k] - 1), axis=0) / (np.sqrt(2 * weights[k]))
        
        fv = np.concatenate([grad_means.flatten(), grad_vars.flatten()])
        return self.power_l2_normalize(fv)
    
    def power_l2_normalize(self, v):
        v = np.multiply(np.sign(v), np.sqrt(np.abs(v)))
        return v / (np.linalg.norm(v) + 1e-8)

class WordSpottingInference:
    def __init__(self):
        self.phoc = PHOC()
        self.fisher = FisherVector()
        self.model = None
        self.use_cuda = torch.cuda.is_available()
    
    def load_model(self, model_path):
   
        print(f" Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f" Model file not found: {model_path}")
            return False
        
        try:
       
            model_data = torch.load(model_path, map_location='cpu', weights_only=False)
            
          
            self.fisher.pca = model_data['fisher_pca']
            self.fisher.gmm = model_data['fisher_gmm']
            self.fisher.vocab_size = model_data['fisher_vocab_size']
            
            self.phoc.levels = model_data['phoc_levels']
            self.phoc.bigrams = model_data['phoc_bigrams']
            
   
            if model_data['model_state_dict'] is not None:
                input_dim = model_data['input_dim']
                output_dim = model_data['output_dim']
                
                print(f"Model dimensions - Input: {input_dim}, Output: {output_dim}")
                
     
                self.model = RobustTorchPHOCModel(input_dim, output_dim)
                
                
                self.model.load_state_dict(model_data['model_state_dict'])
                self.model.eval()
                
                if self.use_cuda:
                    self.model = self.model.cuda()
                    print(f" Model loaded on GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print(" Model loaded on CPU")
                
                print(f" Model architecture: Input({input_dim}) -> 32 -> Output({output_dim})")
            
            print(" Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f" Error loading model: {e}")
            print("🔄 Trying alternative loading method...")
            
            try:
          
                model_data = torch.load(model_path, map_location='cpu')
                
           
                if 'fisher_pca' in model_data:
                    self.fisher.pca = model_data['fisher_pca']
                if 'fisher_gmm' in model_data:
                    self.fisher.gmm = model_data['fisher_gmm']
                if 'fisher_vocab_size' in model_data:
                    self.fisher.vocab_size = model_data['fisher_vocab_size']
                if 'phoc_levels' in model_data:
                    self.phoc.levels = model_data['phoc_levels']
                if 'phoc_bigrams' in model_data:
                    self.phoc.bigrams = model_data['phoc_bigrams']
                

                if 'model_state_dict' in model_data and model_data['model_state_dict'] is not None:
                    input_dim = model_data.get('input_dim', 256)  # Default if not found
                    output_dim = model_data.get('output_dim', 62)  # Default if not found
                    
                    self.model = RobustTorchPHOCModel(input_dim, output_dim)
                    self.model.load_state_dict(model_data['model_state_dict'])
                    self.model.eval()
                    
                    if self.use_cuda:
                        self.model = self.model.cuda()
                    
                    print(" Model loaded with alternative method!")
                    return True
                    
            except Exception as e2:
                print(f" Alternative loading also failed: {e2}")
                return False
    
    def load_and_preprocess_image(self, image_path, target_height=64):

        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f" Could not load image: {image_path}")
                return None
            
            h, w = image.shape
            new_w = max(32, int(w * target_height / h))
            resized = cv2.resize(image, (new_w, target_height))
            

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            
            return enhanced
        except Exception as e:
            print(f" Error preprocessing image {image_path}: {e}")
            return None
    
    def extract_sift_features(self, image):
 
        try:
            sift = cv2.SIFT_create()
            
     
            keypoints = []
            step_size = 12
            for y in range(0, image.shape[0], step_size):
                for x in range(0, image.shape[1], step_size):
                    keypoints.append(cv2.KeyPoint(x, y, step_size))
            
            _, descriptors = sift.compute(image, keypoints)
            
            if descriptors is None or len(descriptors) == 0:
          
                print("⚠️ No SIFT features found, using fallback descriptors")
                descriptors = np.random.randn(20, 128).astype(np.float32)
            
            return descriptors
            
        except Exception as e:
            print(f" Error extracting SIFT features: {e}")
            return np.random.randn(20, 128).astype(np.float32)
    
    def extract_features(self, image_path):

        try:
            image = self.load_and_preprocess_image(image_path)
            if image is None:
                return None
            
            descriptors = self.extract_sift_features(image)
            if descriptors is None:
                return None
            
            fv = self.fisher.transform(descriptors)
            return fv
            
        except Exception as e:
            print(f" Error extracting features from {image_path}: {e}")
            return None
    
    def predict_phoc_from_image(self, image_path):
    
        if self.model is None:
            print(" Model not loaded")
            return None
        
        # Extract features
        fv = self.extract_features(image_path)
        if fv is None:
            print(f" Could not extract features from {image_path}")
            return None
        
      
        self.model.eval()
        with torch.no_grad():
            fv_tensor = torch.FloatTensor(fv).unsqueeze(0)
            
            if self.use_cuda:
                fv_tensor = fv_tensor.cuda()
            
            phoc_pred = self.model(fv_tensor)
            
            if self.use_cuda:
                phoc_pred = phoc_pred.cpu()
            
            predicted_phoc = phoc_pred.numpy()[0]
        
        return predicted_phoc
    
    def query_by_image(self, query_image_path, database_images, top_k=5):
     
        print(f" Querying by image: {query_image_path}")
        
        query_phoc = self.predict_phoc_from_image(query_image_path)
        if query_phoc is None:
            print(" Could not process query image")
            return []
        
        print(" Processing database images...")
   
        database_phocs = {}
        for i, img_path in enumerate(database_images):
            if i % 10 == 0:
                print(f"   Processed {i}/{len(database_images)} images...")
            phoc = self.predict_phoc_from_image(img_path)
            if phoc is not None:
                database_phocs[img_path] = phoc
        
        print(f" Processed {len(database_phocs)} database images")
        
        if not database_phocs:
            print(" No valid database images found")
            return []
        
        # Calculate similarities
        similarities = {}
        query_phoc_flat = query_phoc.flatten()
        
        for img_path, phoc in database_phocs.items():
            phoc_flat = phoc.flatten()
            min_dim = min(len(query_phoc_flat), len(phoc_flat))
            
            # Cosine similarity
            dot_product = np.dot(query_phoc_flat[:min_dim], phoc_flat[:min_dim])
            norm_query = np.linalg.norm(query_phoc_flat[:min_dim])
            norm_db = np.linalg.norm(phoc_flat[:min_dim])
            
            if norm_query > 0 and norm_db > 0:
                sim = dot_product / (norm_query * norm_db)
            else:
                sim = 0
                
            similarities[img_path] = sim
        

        results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return results
    
    def query_by_string(self, query_string, database_images, top_k=5):
   
        print(f" Querying by text: '{query_string}'")
        
        query_phoc = self.phoc(query_string)
        
        print(" Processing database images...")
      
        database_phocs = {}
        for i, img_path in enumerate(database_images):
            if i % 10 == 0:
                print(f"   Processed {i}/{len(database_images)} images...")
            phoc = self.predict_phoc_from_image(img_path)
            if phoc is not None:
                database_phocs[img_path] = phoc
        
        print(f" Processed {len(database_phocs)} database images")
        
        if not database_phocs:
            print(" No valid database images found")
            return []
        
      
        similarities = {}
        query_phoc_flat = query_phoc.flatten()
        
        for img_path, phoc in database_phocs.items():
            phoc_flat = phoc.flatten()
            min_dim = min(len(query_phoc_flat), len(phoc_flat))
            
          
            dot_product = np.dot(query_phoc_flat[:min_dim], phoc_flat[:min_dim])
            norm_query = np.linalg.norm(query_phoc_flat[:min_dim])
            norm_db = np.linalg.norm(phoc_flat[:min_dim])
            
            if norm_query > 0 and norm_db > 0:
                sim = dot_product / (norm_query * norm_db)
            else:
                sim = 0
                
            similarities[img_path] = sim
        
        # Return top matches
        results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return results

def load_sample_images(data_root, max_images=1000):

    image_paths = []
    data_path = Path(data_root)
    
    print("Loading sample images...")
    
 
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    all_image_files = []
    
    for ext in extensions:
        all_image_files.extend(list(data_path.rglob(ext)))
    
    print(f" Found {len(all_image_files)} total images")
    

    subset_files = all_image_files[:max_images]
    for img_path in subset_files:
        image_paths.append(str(img_path))
    
    print(f" Loaded {len(image_paths)} sample images")
    return image_paths

def display_results(results, query_type, query_value):

    if not results:
        print(" No results found")
        return
    
    print(f"\n Top matches for {query_type} '{query_value}':")
    print("=" * 60)
    
    for i, (img_path, similarity) in enumerate(results):
        print(f"{i+1:2d}. {os.path.basename(img_path)}")
        print(f"    Similarity: {similarity:.3f}")
        print(f"    Path: {img_path}")
        print()

def main():
    MODEL_PATH = r"C:\Users\uditr\Downloads\CV Project\Models\realistic_word_spotting_model.pth"
    DATA_ROOT = r"C:\Users\uditr\Downloads\archive\data"
    
    print(" Word Spotting Inference System")
    print("=" * 50)
    

    inference_system = WordSpottingInference()
    

    if not inference_system.load_model(MODEL_PATH):
        print(" Failed to load model. Exiting...")
        return
    

    database_images = load_sample_images(DATA_ROOT, 1000)
    
    if not database_images:
        print(" No database images found. Exiting...")
        return
    
    print("\n Word Spotting Inference Ready!")
    print("=" * 50)
    
    while True:
        print("\n Choose query type:")
        print("1. Query by Image ")
        print("2. Query by Text ")
        print("3. Exit")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "1":
            # Query by Image
            query_path = input("Enter path to query image: ").strip()
            if not os.path.exists(query_path):
                print(" Image file not found!")
                continue
            
            print(f"\n Searching for images similar to: {os.path.basename(query_path)}")
            results = inference_system.query_by_image(query_path, database_images, top_k=5)
            
            display_results(results, "image", os.path.basename(query_path))
                
        elif choice == "2":
         
            query_text = input("Enter text to search for: ").strip()
            if not query_text:
                print(" Please enter valid text!")
                continue
            
            print(f"\n Searching for images matching: '{query_text}'")
            results = inference_system.query_by_string(query_text, database_images, top_k=5)
            
            display_results(results, "text", query_text)
                
        elif choice == "3":
            print("👋 Exiting...")
            break
            
        else:
            print(" Invalid choice! Please enter 1, 2, or 3")

if __name__ == "__main__":
    main()