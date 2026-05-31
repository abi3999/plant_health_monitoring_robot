import torch
import clip
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, json, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
DATASET_DIR = r"C:\Users\abhis\Desktop\plantvillage dataset\color"
EPOCHS      = 10          # Increased from 5 for better convergence
BATCH_SIZE  = 32          # Doubled from 16 (better GPU utilization)
LR          = 1e-6        # Increased from 1e-7 (faster convergence with larger batch)
WEIGHT_DECAY = 0.05       # Slightly increased regularization
GRAD_CLIP   = 1.0         # Relaxed from 0.1 (allows bigger steps)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization & logging
SAVE_DIR    = "training_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def label_to_prompt(label):
    """Convert raw label to natural language prompt for CLIP."""
    parts     = label.split("___")
    plant     = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "healthy"
    if "healthy" in condition.lower():
        return f"a healthy {plant} leaf with no disease"
    return f"a {plant} leaf with {condition} disease"


class PlantDataset(Dataset):
    """
    Enhanced dataset with:
    - Better error handling
    - Optional data augmentation
    - Consistent preprocessing
    """
    def __init__(self, root_dir, preprocess, split="train", val_ratio=0.2, augment=False):
        self.preprocess   = preprocess
        self.split        = split
        self.augment      = augment and (split == "train")
        self.classes      = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        all_samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_samples.append((os.path.join(cls_dir, fname), cls))

        random.seed(42)
        random.shuffle(all_samples)
        split_idx    = int(len(all_samples) * (1 - val_ratio))
        self.samples = all_samples[:split_idx] if split == "train" else all_samples[split_idx:]

        # Simple augmentation: random horizontal flip
        self.flip = lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() > 0.5 else img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.augment:
                img = self.flip(img)
            image = self.preprocess(img)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            image = torch.zeros(3, 224, 224)
        return image, self.class_to_idx[label]


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(history, save_path):
    """Plot and save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CLIP Plant Disease Classification - Training Metrics', fontsize=14, fontweight='bold')

    epochs = range(1, len(history['train_acc']) + 1)

    # Accuracy curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    ax.plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # Loss curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy comparison bar chart
    ax = axes[1, 0]
    x = np.arange(len(epochs))
    width = 0.35
    ax.bar(x - width/2, history['train_acc'], width, label='Train', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, history['val_acc'], width, label='Val', color='coral', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Train vs Val Accuracy (Bar)')
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Best model indicator
    ax = axes[1, 1]
    ax.axis('off')
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    info_text = f"""TRAINING SUMMARY

Best Validation Accuracy: {best_acc:.2f}%
Best Epoch: {best_epoch}/{len(epochs)}

Final Train Accuracy: {history['train_acc'][-1]:.2f}%
Final Val Accuracy: {history['val_acc'][-1]:.2f}%

Trainable Parameters: ~5.2M / 86M
Learning Rate: {LR}
Batch Size: {BATCH_SIZE}

Model saved: clip_plant_best.pt
Labels saved: clip_labels.json
    """
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix with per-class metrics."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row (per-class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Confusion Matrix Analysis', fontsize=14, fontweight='bold')

    # Raw counts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 8})
    ax1.set_title('Confusion Matrix (Raw Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)

    # Normalized (percentages)
    ax2 = axes[1]
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='RdYlGn', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, annot_kws={"size": 8})
    ax2.set_title('Confusion Matrix (Normalized by Row)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()

    # Also print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    return cm, cm_norm


def plot_per_class_accuracy(cm_norm, class_names, save_path):
    """Plot per-class accuracy bar chart."""
    per_class_acc = np.diag(cm_norm) * 100

    fig, ax = plt.subplots(figsize=(14, max(6, len(class_names) * 0.4)))

    colors = ['#4CAF50' if acc > 80 else '#FF9800' if acc > 60 else '#F44336' for acc in per_class_acc]
    bars = ax.barh(range(len(class_names)), per_class_acc, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_title('Per-Class Accuracy on Validation Set', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim([0, 105])
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Add average line
    avg_acc = np.mean(per_class_acc)
    ax.axvline(avg_acc, color='blue', linestyle='--', linewidth=2, label=f'Average: {avg_acc:.1f}%')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Per-class accuracy saved to: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    start_time = datetime.now()
    print("="*60)
    print("  ENHANCED CLIP PLANT DISEASE CLASSIFICATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\nLoading CLIP ViT-B/16...")
    model, preprocess = clip.load("ViT-B/16", device=DEVICE)

    # CRITICAL: convert to float32
    model = model.float()
    print("Model converted to float32 ✓")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the last visual transformer block + projection
    for name, param in model.named_parameters():
        if any(x in name for x in [
            "visual.transformer.resblocks.11",
            "visual.ln_post",
            "visual.proj",
        ]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_ds = PlantDataset(DATASET_DIR, preprocess, split="train", augment=True)
    val_ds   = PlantDataset(DATASET_DIR, preprocess, split="val", augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False
    )

    num_classes = len(train_ds.classes)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Classes: {num_classes}")

    # Save labels to file
    with open("clip_labels.json", "w") as f:
        json.dump({
            "classes": train_ds.classes,
            "prompts": [label_to_prompt(c) for c in train_ds.classes]
        }, f, indent=2)
    print("Labels saved to clip_labels.json ✓")

    # ── Optimizer & Loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()

    # Tokenise all class prompts once
    all_prompts = [label_to_prompt(c) for c in train_ds.classes]
    text_tokens = clip.tokenize(all_prompts, truncate=True).to(DEVICE)

    # ── Training History ──────────────────────────────────────────────────────
    history = {
        'train_acc': [], 'train_loss': [],
        'val_acc': [], 'val_loss': [],
        'nan_batches': []
    }
    best_val_acc = 0.0
    best_epoch = 0
    patience = 3  # Early stopping patience
    patience_counter = 0

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()

        # Recompute text features at start of each epoch
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = nn.functional.normalize(text_features, dim=-1)

        total_loss = 0.0
        correct    = 0
        total      = 0
        nan_count  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            image_features = model.encode_image(images)
            image_features = nn.functional.normalize(image_features, dim=-1)

            logits = 100.0 * (image_features @ text_features.T)
            loss   = loss_fn(logits, labels)

            # Skip bad batches
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(dim=-1) == labels).sum().item()
            total      += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total*100:.1f}%'
            })

        train_acc = correct / total * 100 if total > 0 else 0.0
        avg_loss  = total_loss / max(len(train_loader) - nan_count, 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total   = 0
        val_loss_total = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = nn.functional.normalize(text_features, dim=-1)

            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ", leave=False)
            for images, labels in pbar_val:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                image_features = model.encode_image(images)
                image_features = nn.functional.normalize(image_features, dim=-1)

                logits = 100.0 * (image_features @ text_features.T)
                loss = loss_fn(logits, labels)

                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                val_loss_total += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar_val.set_postfix({'acc': f'{val_correct/val_total*100:.1f}%'})

        val_acc = val_correct / val_total * 100
        val_loss = val_loss_total / len(val_loader)

        # Store history
        history['train_acc'].append(train_acc)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['nan_batches'].append(nan_count)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train accuracy : {train_acc:.2f}%")
        print(f"  Val accuracy   : {val_acc:.2f}%")
        print(f"  Train loss     : {avg_loss:.4f}")
        print(f"  Val loss       : {val_loss:.4f}")
        print(f"  NaN batches    : {nan_count}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "clip_plant_best.pt")
            print(f"  ✓ NEW BEST MODEL SAVED — {val_acc:.2f}% (epoch {best_epoch})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  (No improvement — patience {patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Training curves
    curves_path = os.path.join(SAVE_DIR, "training_curves.png")
    plot_training_curves(history, curves_path)

    # 2. Confusion matrix (using best model)
    print("\nReloading best model for final evaluation...")
    model.load_state_dict(torch.load("clip_plant_best.pt"))
    model.eval()

    all_preds_final = []
    all_labels_final = []

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = nn.functional.normalize(text_features, dim=-1)

        for images, labels in tqdm(val_loader, desc="Final evaluation"):
            images = images.to(DEVICE)
            image_features = model.encode_image(images)
            image_features = nn.functional.normalize(image_features, dim=-1)
            logits = 100.0 * (image_features @ text_features.T)
            preds = logits.argmax(dim=-1)
            all_preds_final.extend(preds.cpu().numpy())
            all_labels_final.extend(labels.cpu().numpy())

    # Confusion matrix
    cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
    cm, cm_norm = plot_confusion_matrix(all_labels_final, all_preds_final, train_ds.classes, cm_path)

    # Per-class accuracy
    per_class_path = os.path.join(SAVE_DIR, "per_class_accuracy.png")
    plot_per_class_accuracy(cm_norm, train_ds.classes, per_class_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"Best val accuracy : {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Final train acc   : {history['train_acc'][-1]:.2f}%")
    print(f"Final val acc     : {history['val_acc'][-1]:.2f}%")
    print(f"Total epochs      : {len(history['train_acc'])}/{EPOCHS}")
    print(f"Training time     : {duration}")
    print(f"Model saved       : clip_plant_best.pt")
    print(f"Labels saved      : clip_labels.json")
    print(f"Visualizations    : {SAVE_DIR}/")
    print("="*60)

    # Save training history as JSON
    history_path = os.path.join(SAVE_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc'],
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'duration_seconds': duration.total_seconds(),
            'config': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'lr': LR,
                'weight_decay': WEIGHT_DECAY,
                'grad_clip': GRAD_CLIP
            }
        }, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")