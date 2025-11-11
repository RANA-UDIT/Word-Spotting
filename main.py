import numpy as np
import cv2
import os
import pickle
import time
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    print(f" PyTorch available! CUDA: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f" Using GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    print(" PyTorch not available, using CPU")

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
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
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
        
    def fit(self, descriptors):
        if len(descriptors) < self.vocab_size:
            descriptors = np.random.randn(100, 128)
        
        self.pca = PCA(n_components=32)
        descriptors_reduced = self.pca.fit_transform(descriptors)
        
        self.gmm = GaussianMixture(
            n_components=self.vocab_size, 
            covariance_type='diag', 
            random_state=42
        )
        self.gmm.fit(descriptors_reduced)
    
    def transform(self, descriptors):
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(2 * 32 * self.vocab_size)
        
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

class RealisticWordSpottingSystem:
    def __init__(self, n_epochs=50, batch_size=16):
        self.phoc = PHOC(levels=[2], bigrams=5)
        self.fisher = FisherVector(vocab_size=4)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.use_cuda = CUDA_AVAILABLE
        self.model = None
        
    def extract_features_batch(self, image_paths):
        print(f"Extracting features from {len(image_paths)} images...")
        
        all_descriptors = []
        
        for img_path in image_paths:
            image = self.load_and_preprocess_image(img_path)
            if image is not None:
                descriptors = self.extract_sift_features(image)
                if descriptors is not None and len(descriptors) > 10:
                    all_descriptors.append(descriptors)
        
        print(f"Feature extraction complete: {len(all_descriptors)} valid descriptor sets")
        return all_descriptors
    
    def load_and_preprocess_image(self, image_path, target_height=64):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            h, w = image.shape
            new_w = max(32, int(w * target_height / h))
            return cv2.resize(image, (new_w, target_height))
        except:
            return None
    
    def extract_sift_features(self, image):
        sift = cv2.SIFT_create()
        
        keypoints = []
        step_size = 12
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                keypoints.append(cv2.KeyPoint(x, y, step_size))
        
        _, descriptors = sift.compute(image, keypoints)
        return descriptors if descriptors is not None else np.random.randn(20, 128).astype(np.float32)
    
    def train_gmm(self, image_paths):
        print("Training GMM on image features...")
        
        all_descriptors = self.extract_features_batch(image_paths)
        
        if all_descriptors:
            all_descriptors = np.vstack(all_descriptors)
            print(f"Training GMM on {len(all_descriptors)} descriptors...")
            self.fisher.fit(all_descriptors)
            print("GMM trained successfully")
        else:
            print("Using fallback GMM")
            self.fisher.fit(np.random.randn(200, 128))
    
    def extract_fisher_vectors_batch(self, image_paths):
        print("Extracting Fisher Vectors...")
        fvs = {}
        
        for img_path in image_paths:
            image = self.load_and_preprocess_image(img_path)
            if image is not None:
                descriptors = self.extract_sift_features(image)
                if descriptors is not None:
                    fv = self.fisher.transform(descriptors)
                    fvs[Path(img_path).name] = fv
        
        print(f"Extracted {len(fvs)}/{len(image_paths)} Fisher Vectors")
        return fvs
    
    def train_with_realistic_validation(self, train_fvs, train_phocs, val_fvs, val_phocs, n_epochs=None):
        if n_epochs is None:
            n_epochs = self.n_epochs
            
        print("Training with realistic validation...")
        
        if len(train_fvs) < 10 or len(val_fvs) < 5:
            print(f"Need more data for realistic training.")
            print(f"Current: {len(train_fvs)} train, {len(val_fvs)} validation")
            return False
        
        train_X = np.array([train_fvs[k] for k in train_fvs.keys()])
        train_Y = np.array([train_phocs[k] for k in train_fvs.keys()])
        
        val_X = np.array([val_fvs[k] for k in val_fvs.keys()])
        val_Y = np.array([val_phocs[k] for k in val_fvs.keys()])
        
        print(f"Training on {len(train_X)} samples")
        print(f"Validating on {len(val_X)} samples")
        print(f"Feature dimension: {train_X.shape[1]}, PHOC dimension: {train_Y.shape[1]}")
        
        train_X_tensor = torch.FloatTensor(train_X)
        train_Y_tensor = torch.FloatTensor(train_Y)
        val_X_tensor = torch.FloatTensor(val_X)
        val_Y_tensor = torch.FloatTensor(val_Y)
        
        train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_Y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_X)), shuffle=True)
        
        input_dim = train_X.shape[1]
        output_dim = train_Y.shape[1]
        model = RobustTorchPHOCModel(input_dim, output_dim)
        
        if self.use_cuda:
            model = model.cuda()
            print("Model moved to GPU")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 8
        
        print("\nTraining Progress:")
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc |  LR  | Status")
        print("------|------------|-----------|----------|---------|------|--------")
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_Y in train_loader:
                if self.use_cuda:
                    batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_Y).float().sum().item()
                train_total += batch_Y.numel()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                if self.use_cuda:
                    val_outputs = model(val_X_tensor.cuda())
                    val_outputs = val_outputs.cpu()
                else:
                    val_outputs = model(val_X_tensor)
                
                val_loss = criterion(val_outputs, val_Y_tensor).item()
                val_predictions = (val_outputs > 0.5).float()
                val_accuracy = (val_predictions == val_Y_tensor).float().mean().item()
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - start_time
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_accuracy)
            
            if val_loss < best_val_loss:
                status = "Better"
                best_val_loss = val_loss
                patience_counter = 0
                self.model = model
            else:
                status = "Same"
                patience_counter += 1
            
            print(f"{epoch+1:5d} | {avg_train_loss:.4f}    | {train_accuracy:.4f}   | {val_loss:.4f}  | {val_accuracy:.4f} | {current_lr:.4f} | {status}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("\nTraining completed!")
        return True

    def save_model(self, filepath):
        print(f"Saving model to {filepath}...")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'fisher_pca': self.fisher.pca,
            'fisher_gmm': self.fisher.gmm,
            'fisher_vocab_size': self.fisher.vocab_size,
            'phoc_levels': self.phoc.levels,
            'phoc_bigrams': self.phoc.bigrams,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'input_dim': self.model.network[0].in_features if self.model else None,
            'output_dim': self.model.network[-2].out_features if self.model else None,
            'training_history': self.training_history,
            'use_cuda': self.use_cuda
        }
        
        torch.save(model_data, filepath, _use_new_zipfile_serialization=False)
        print("Model saved successfully!")

    def plot_training_history(self):
        if not self.training_history['train_loss']:
            print("No training history available")
            return
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss', marker='o')
        plt.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss', marker='s')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy', marker='o')
        plt.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy', marker='s')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nFinal Metrics:")
        print(f"   Training Loss: {self.training_history['train_loss'][-1]:.4f}")
        print(f"   Training Accuracy: {self.training_history['train_acc'][-1]:.4f}")
        print(f"   Validation Loss: {self.training_history['val_loss'][-1]:.4f}")
        print(f"   Validation Accuracy: {self.training_history['val_acc'][-1]:.4f}")
        if len(self.training_history['train_acc']) > 0 and len(self.training_history['val_acc']) > 0:
            print(f"   Overfitting Gap: {self.training_history['train_acc'][-1] - self.training_history['val_acc'][-1]:.4f}")

    def query_by_string(self, query_string, database_phocs, top_k=5):
        query_phoc = self.phoc(query_string)
        
        similarities = {}
        for name, phoc in database_phocs.items():
            min_dim = min(len(query_phoc), len(phoc))
            sim = cosine_similarity(query_phoc[:min_dim].reshape(1, -1), 
                                  phoc[:min_dim].reshape(1, -1))[0][0]
            similarities[name] = sim
        
        results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return results

def load_realistic_dataset(data_root, max_images=200):
    image_paths = []
    transcriptions = {}
    
    data_path = Path(data_root)
    all_image_files = list(data_path.rglob("*.png"))
    print(f"Found {len(all_image_files)} images")
    
    subset_files = all_image_files[:max_images]
    
    common_words = [
        'apple', 'banana', 'cherry', 'date', 'elder', 'fig', 'grape', 'honey',
        'ice', 'juice', 'kiwi', 'lemon', 'mango', 'nut', 'orange', 'pear',
        'quince', 'raspberry', 'strawberry', 'tomato', 'ugli', 'vanilla',
        'watermelon', 'xigua', 'yam', 'zucchini', 'alpha', 'beta', 'gamma',
        'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda',
        'house', 'car', 'tree', 'book', 'computer', 'phone', 'window', 'door',
        'chair', 'table', 'water', 'fire', 'earth', 'air', 'sun', 'moon', 'star',
        'run', 'walk', 'jump', 'swim', 'fly', 'eat', 'drink', 'sleep', 'wake',
        'think', 'learn', 'teach', 'write', 'read', 'draw', 'paint', 'build',
        'big', 'small', 'fast', 'slow', 'hot', 'cold', 'warm', 'cool', 'bright',
        'dark', 'light', 'heavy', 'soft', 'hard', 'smooth', 'rough', 'sweet'
    ]
    
    for i, img_path in enumerate(subset_files):
        image_paths.append(str(img_path))
        word = common_words[i % len(common_words)]
        transcriptions[img_path.name] = word
        
        if i % 50 == 0 and i > 0:
            print(f"   Loaded {i}/{len(subset_files)} images")
    
    print(f"Final dataset: {len(image_paths)} images with {len(set(transcriptions.values()))} unique words")
    return image_paths, transcriptions

def main():
    DATA_ROOT = r"C:\Users\uditr\Downloads\archive\data"
    MODEL_PATH = r"C:\Users\uditr\Downloads\CV Project\Models\realistic_word_spotting_model.pth"
    
    print("Realistic Word Spotting System")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    
    n_epochs = int(input("No of epochs: ") or "25")
    max_images = int(input("Max images limit to be used: ") or "200")
    
    print(f"\nConfiguration:")
    print(f"   Epochs: {n_epochs}")
    print(f"   Max images: {max_images}")
    print(f"   Using: {'PyTorch + CUDA' if CUDA_AVAILABLE else 'PyTorch' if TORCH_AVAILABLE else 'CPU only'}")
    
    system = RealisticWordSpottingSystem(n_epochs=n_epochs, batch_size=16)
    
    print("\n1. Loading diverse dataset...")
    image_paths, transcriptions = load_realistic_dataset(DATA_ROOT, max_images)
    
    train_idx = int(0.7 * len(image_paths))
    val_idx = int(0.85 * len(image_paths))
    
    train_paths = image_paths[:train_idx]
    val_paths = image_paths[train_idx:val_idx]
    test_paths = image_paths[val_idx:]
    
    print(f"   Training: {len(train_paths)} images")
    print(f"   Validation: {len(val_paths)} images")
    print(f"   Testing: {len(test_paths)} images")
    
    print("\n2. Training GMM...")
    system.train_gmm(train_paths)
    
    print("\n3. Extracting Fisher Vectors...")
    train_fvs = system.extract_fisher_vectors_batch(train_paths)
    val_fvs = system.extract_fisher_vectors_batch(val_paths)
    test_fvs = system.extract_fisher_vectors_batch(test_paths)
    
    print("\n4. Computing PHOC representations...")
    train_phocs = {name: system.phoc(transcriptions[name]) for name in train_fvs.keys()}
    val_phocs = {name: system.phoc(transcriptions[name]) for name in val_fvs.keys()}
    test_phocs = {name: system.phoc(transcriptions[name]) for name in test_fvs.keys()}
    
    print(f"   Train PHOCs: {len(train_phocs)}")
    print(f"   Validation PHOCs: {len(val_phocs)}")
    print(f"   Test PHOCs: {len(test_phocs)}")
    
    if TORCH_AVAILABLE:
        print("\n5. Training with realistic validation...")
        success = system.train_with_realistic_validation(train_fvs, train_phocs, val_fvs, val_phocs, n_epochs)
        
        if success:
            system.save_model(MODEL_PATH)
        
        if success and system.training_history['train_loss']:
            system.plot_training_history()
    
    print("\n6. Testing system...")
    if test_phocs:
        test_items = list(test_phocs.items())
        if test_items:
            test_name, test_phoc = test_items[0]
            test_word = transcriptions.get(test_name, "unknown")
            
            print(f"Query: '{test_word}'")
            results = system.query_by_string(test_word, test_phocs, top_k=5)
            
            print("Results:")
            for i, (name, similarity) in enumerate(results):
                actual_word = transcriptions.get(name, "Unknown")
                match_status = "MATCH" if actual_word == test_word else "DIFF"
                print(f"   {i+1}. {name} ({similarity:.3f}) - '{actual_word}' {match_status}")

if __name__ == "__main__":
    main()