import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
import os
import json
import pandas as pd
from collections import Counter
import gc

class SigLIPInference:
    def __init__(self, model_name="google/siglip-large-patch16-384", device=None):
        """Initialize the SigLIP model and processor."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def cleanup(self):
        """Frees GPU/CPU memory by deleting the model and clearing cache."""
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_data(self, image_dir, image_list_file, caption_file, is_json=False):
        """Load images and captions from files."""
        # Load image paths
        with open(image_list_file, "r") as f:
            image_filenames = [line.strip() for line in f if line.strip()]
        image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]
        
        # Load captions
        if is_json:
            with open(caption_file, "r") as f:
                captions = json.load(f)
        else:
            with open(caption_file, "r") as f:
                captions = [line.strip() for line in f if line.strip()]
        
        return image_paths, captions

    def process_batch(self, image_batch, caption_batch):
        """Process a batch of images and captions."""
        inputs = self.processor(
            text=caption_batch,
            images=image_batch,
            return_tensors="pt",
            padding='max_length',
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.logits_per_image

    def predict_multi_label(self, image_paths, captions, image_batch_size=256, caption_batch_size=512, threshold=0.95, k = 5, method="sigmoid"):
        """Generate multi-label predictions for images."""
        num_images = len(image_paths)
        num_captions = len(captions)
        row_indices, col_indices = [], []

        for image_start in tqdm(range(0, num_images, image_batch_size), desc="Processing Images"):
            image_end = min(image_start + image_batch_size, num_images)
            batch_paths = image_paths[image_start:image_end]

            # Load and preprocess images
            image_batch = []
            valid_indices = []
            for i, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    image_batch.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"⚠️ Failed to load image {path}: {e}")
                    continue

            if not image_batch:
                continue

            # Process captions in chunks
            all_logits = []
            for cap_start in range(0, num_captions, caption_batch_size):
                cap_end = min(cap_start + caption_batch_size, num_captions)
                caption_batch = captions[cap_start:cap_end]
                logits = self.process_batch(image_batch, caption_batch)
                all_logits.append(logits.cpu())

            # Combine logits and get predictions
            full_logits = torch.cat(all_logits, dim=1)
            if method == 'softmax':
                probs = torch.softmax(full_logits, dim=1)

                # Select predictions based on cumulative probability threshold
                for i, prob in enumerate(probs):
                    sorted_probs, sorted_indices = torch.sort(prob, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    num_selected = min((cumulative_probs < threshold).sum().item() + 1, k)
                    selected_indices = sorted_indices[:num_selected]

                    pred = torch.zeros_like(prob)
                    pred[selected_indices] = 1
                    
                    # Get indices where prediction is 1
                    pred_indices = torch.where(pred == 1)[0].numpy()
                    for pred_idx in pred_indices:
                        row_indices.append(image_start + valid_indices[i])
                        col_indices.append(pred_idx)
            else:
                probs = torch.sigmoid(full_logits)

                topk_indices = torch.topk(probs, k=k, dim=1).indices
                pred_matrix = torch.zeros_like(probs).scatter(1, topk_indices, 1).int().cpu().numpy()

                row_index, col_index = np.where(pred_matrix == 1)
                row_indices.extend(row_index + image_start)
                col_indices.extend(col_index)

        return row_indices, col_indices, num_images, num_captions

    def predict_single_label(self, image_paths, captions, image_batch_size=128, caption_batch_size=1024, method='sigmoid'):
        """Generate single-label predictions for images."""
        num_images = len(image_paths)
        num_captions = len(captions)
        row_indices, col_indices = [], []

        for image_start in tqdm(range(0, num_images, image_batch_size), desc="Processing Images"):
            batch_paths = image_paths[image_start:image_start + image_batch_size]
            
            # Load and preprocess images
            image_batch = []
            valid_indices = []
            for i, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    image_batch.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"⚠️ Failed to load image {path}: {e}")
                    continue

            if not image_batch:
                continue

            # Process captions in chunks
            all_logits = []
            for cap_start in range(0, num_captions, caption_batch_size):
                cap_end = min(cap_start + caption_batch_size, num_captions)
                caption_batch = captions[cap_start:cap_end]
                logits = self.process_batch(image_batch, caption_batch)
                all_logits.append(logits.cpu())

            # Get predictions
            full_logits = torch.cat(all_logits, dim=1)
            if method == 'softmax':
                probs = torch.softmax(full_logits, dim=1)
            else:
                probs = torch.sigmoid(full_logits)
            top1 = torch.argmax(probs, dim=1).numpy()

            for i, cap_idx in enumerate(top1):
                row_indices.append(image_start + valid_indices[i])
                col_indices.append(cap_idx)

        return row_indices, col_indices, num_images, num_captions

    def save_predictions(self, row_indices, col_indices, num_images, num_captions, output_file):
        """Save predictions as a sparse matrix."""
        data = np.ones(len(row_indices), dtype=int)
        sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_images, num_captions))
        save_npz(output_file, sparse_matrix)
        print(f"✅ Saved predictions to {output_file}")

def create_submission_csv(test1_npz, test2_npz, num_images_test1, num_images_test2, output_file):
    """Create submission CSV from prediction matrices."""
    # Load sparse matrices
    test1_matrix = load_npz(test1_npz)
    test2_matrix = load_npz(test2_npz)
    
    image_ids = []
    class_ids = []

    # Process Test 1 predictions
    coo1 = test1_matrix.tocoo()
    for img_id in range(num_images_test1):
        mask = (coo1.row == img_id)
        selected_cols = coo1.col[mask]
        preds = "-".join(str(c) for c in selected_cols) if selected_cols.size > 0 else ""
        image_ids.append(img_id)
        class_ids.append(preds)
    
    # Process Test 2 predictions
    coo2 = test2_matrix.tocoo()
    for img_id in range(num_images_test2):
        new_img_id = num_images_test1 + img_id
        mask = (coo2.row == img_id)
        selected_cols = coo2.col[mask]
        preds = "-".join(str(c) for c in selected_cols) if selected_cols.size > 0 else ""
        image_ids.append(new_img_id)
        class_ids.append(preds)

    # Create and save DataFrame
    df = pd.DataFrame({
        "image_id": image_ids,
        "class_ids": class_ids
    })
    df.to_csv(output_file, index=False)
    print(f"✅ Final CSV saved: {len(df)} rows, {len(df)*2+1} total lines including header.")

def ensemble_submissions(file_paths, tiebreaker_file_path):

    # Load all submission files
    dfs = [pd.read_csv(f) for f in file_paths]
    tiebreaker_df = pd.read_csv(tiebreaker_file_path)
    
    # Verify all have same length and IDs
    for df in dfs[1:] + [tiebreaker_df]:
        assert len(df) == len(dfs[0]), "All files must have same number of rows"
        assert (df['image_id'] == dfs[0]['image_id']).all(), "All files must have same IDs in same order"
    
    # Initialize result dataframe
    result = pd.DataFrame()
    result['image_id'] = dfs[0]['image_id']
    
    # Process Test 1 (rows 0-3821) - Multi-label
    test1_labels = []
    for i in range(3823):
        # Collect all labels from all files for this row
        label_counts = Counter()
        for df in dfs:
            labels = str(df.iloc[i, 1]).split('-')
            for label in labels:
                if label != 'nan':
                    label_counts[label] += 1
        
        # Keep only labels that appear in more than one file
        selected_labels = [label for label, count in label_counts.items() if count > 1]
        
        # If no labels appear multiple times, use tiebreaker's labels
        if not selected_labels:
            tie_labels = str(tiebreaker_df.iloc[i, 1]).split('-')
            selected_labels = [label for label in tie_labels if label != 'nan']
        
        # Sort numerically and join
        merged_labels = '-'.join(sorted(selected_labels, key=lambda x: int(x))) if selected_labels else ''
        test1_labels.append(merged_labels)
    
    # Process Test 2 (rows 3823-end) - Single-label with majority voting
    test2_labels = []
    for i in range(3823, len(dfs[0])):
        # Get predictions from all files
        predictions = [str(df.iloc[i, 1]) for df in dfs]
        
        # Count occurrences
        counts = Counter(predictions)
        most_common = counts.most_common()
        
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            # Clear majority exists
            chosen = most_common[0][0]
        else:
            # Tie - use the prediction from tiebreaker file
            chosen = str(tiebreaker_df.iloc[i, 1])
        
        test2_labels.append(chosen)
    
    # Combine both test sets
    result['class_ids'] = test1_labels + test2_labels
    
    return result

def main():
    # Test 1 configuration
    test1_config = {
        "image_dir": "Test1/imgs",
        "image_list_file": "Test1/images.txt",
        "caption_file": "Test1/captions.txt",
        "is_json": False
    }

    # Test 2 configuration
    test2_config = {
        "image_dir": "Test2/imgs",
        "image_list_file": "Test2/images.txt",
        "caption_file": "Test2/captions.json",
        "is_json": True
    }

    print("Model 1: siglip-l16-384")
    model = SigLIPInference()

    # Process Test 1 (multi-label)
    print("Processing Test 1...")
    image_paths, captions = model.load_data(**{k: v for k, v in test1_config.items() if k != 'is_json'})
    row_indices, col_indices, num_images, num_captions = model.predict_multi_label(image_paths, captions, threshold=0.95, k = 5, method='softmax')
    model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions.npz')

    # Process Test 2 (single-label)
    print("Processing Test 2...")
    image_paths, captions = model.load_data(**{k: v for k, v in test2_config.items() if k != 'is_json'})
    row_indices, col_indices, num_images, num_captions = model.predict_single_label(image_paths, captions, method='softmax')
    model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions2.npz')

    # Create final submission
    create_submission_csv(
        'predictions.npz',
        'predictions2.npz',
        num_images_test1=3823,
        num_images_test2=3253,
        output_file="siglip-l16-384.csv"
    )

    print("Model 2: siglip-l16-384-v2")

    # Process Test 1 (multi-label)
    print("Processing Test 1...")
    image_paths, captions = model.load_data(**{k: v for k, v in test1_config.items() if k != 'is_json'})
    row_indices, col_indices, num_images, num_captions = model.predict_multi_label(image_paths, captions, k = 5, method='sigmoid')
    model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions.npz')

    # # Process Test 2 (single-label)
    # print("Processing Test 2...")
    # image_paths, captions = model.load_data(**{k: v for k, v in test2_config.items() if k != 'is_json'})
    # row_indices, col_indices, num_images, num_captions = model.predict_single_label(image_paths, captions)
    # model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions2.npz')

    # Create final submission
    create_submission_csv(
        'predictions.npz',
        'predictions2.npz',
        num_images_test1=3823,
        num_images_test2=3253,
        output_file="siglip-l16-384-v2.csv"
    )

    print("Model 3: siglip-l16-384-th0.98")

    # Process Test 1 (multi-label)
    print("Processing Test 1...")
    image_paths, captions = model.load_data(**{k: v for k, v in test1_config.items() if k != 'is_json'})
    row_indices, col_indices, num_images, num_captions = model.predict_multi_label(image_paths, captions, threshold = 0.98, k = 7, method='softmax')
    model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions.npz')

    # Process Test 2 (single-label)
    # print("Processing Test 2...")
    # image_paths, captions = model.load_data(**{k: v for k, v in test2_config.items() if k != 'is_json'})
    # row_indices, col_indices, num_images, num_captions = model.predict_single_label(image_paths, captions)
    # model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions2.npz')

    # Create final submission
    create_submission_csv(
        'predictions.npz',
        'predictions2.npz',
        num_images_test1=3823,
        num_images_test2=3253,
        output_file="siglip-l16-384-th0.98.csv"
    )

    model.cleanup()

    print("Baseline")
    model = SigLIPInference(model_name="google/siglip-base-patch16-512")

    # Process Test 1 (multi-label)
    print("Processing Test 1...")
    image_paths, captions = model.load_data(**{k: v for k, v in test1_config.items() if k != 'is_json'})
    row_indices, col_indices, num_images, num_captions = model.predict_multi_label(image_paths, captions, k = 5, method='sigmoid')
    model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions.npz')

    # Process Test 2 (single-label)
    print("Processing Test 2...")
    image_paths, captions = model.load_data(**{k: v for k, v in test2_config.items() if k != 'is_json'})
    row_indices, col_indices, num_images, num_captions = model.predict_single_label(image_paths, captions, method='sigmoid')
    model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions2.npz')

    # Create final submission
    create_submission_csv(
        'predictions.npz',
        'predictions2.npz',
        num_images_test1=3823,
        num_images_test2=3253,
        output_file="baseline.csv"
    )

    model.cleanup()

    print("Model 5: so400m")
    model = SigLIPInference(model_name="google/siglip-so400m-patch14-384")

    # Process Test 1 (multi-label)
    print("Processing Test 1...")
    image_paths, captions = model.load_data(**{k: v for k, v in test1_config.items() if k != 'is_json'})
    row_indices, col_indices, num_images, num_captions = model.predict_multi_label(image_paths, captions, k = 6, method='sigmoid')
    model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions.npz')

    # Process Test 2 (single-label)
    print("Processing Test 2...")
    image_paths, captions = model.load_data(**{k: v for k, v in test2_config.items() if k != 'is_json'})
    row_indices, col_indices, num_images, num_captions = model.predict_single_label(image_paths, captions, method="sigmoid")
    model.save_predictions(row_indices, col_indices, num_images, num_captions, 'predictions2.npz')

    # Create final submission
    create_submission_csv(
        'predictions.npz',
        'predictions2.npz',
        num_images_test1=3823,
        num_images_test2=3253,
        output_file="so400m.csv"
    )

    files_to_ensemble = ['siglip-l16-384.csv', 'siglip-l16-384-v2.csv', 'siglip-l16-384-th0.98.csv', 'so400m.csv','baseline.csv']
    tiebreaker_file = 'siglip-l16-384.csv'  # File to use when votes are tied
    ensembled = ensemble_submissions(files_to_ensemble, tiebreaker_file)
    ensembled.to_csv('ensemble.csv', index=False)

if __name__ == "__main__":
    main() 