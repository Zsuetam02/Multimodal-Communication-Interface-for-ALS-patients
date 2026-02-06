import os
import random
import shutil
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from DataLoaderv2Cl import FusionDataset
from ModelConfiguratorCl import FusionNet

# USER CONFIG
ON_CLUSTER = os.path.exists("/scratch/scratch-hdd/4all/mskrzypczyk")
if ON_CLUSTER:
    DATA_PATH = "/scratch/scratch-hdd/4all/mskrzypczyk/ExportAppViewer/PythonExported"
    OUT_DIR = "/scratch/scratch-hdd/4all/mskrzypczyk/KFoldResults"
else:
    DATA_PATH = "C:/bachelorProject/ExportAppViewer/PythonExported"
    OUT_DIR = "./KFoldResults"

os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS = 10
SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (256, 64)
WINDOW_SIZE = 48
INTERP_FACTOR = 4
MAX_ANNOTATIONS = None
FEATURE_DIM = 14
MODE = "fusion"              # "fusion", "video", "eog"
PATIENCE = 25
NUM_WORKERS = 4 if os.name != "nt" else 0

CLASS_NAMES = ["up", "down", "left", "right", "closed"]
NUM_CLASSES = len(CLASS_NAMES)

# Determinism
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE} | K={N_SPLITS} folds | Mode: {MODE}")

# Helper: find all mat files in folder (filenames only)
all_mat_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".mat")])
if len(all_mat_files) == 0:
    raise RuntimeError(f"No .mat files found in {DATA_PATH}")

# Optionally limit by MAX_ANNOTATIONS
if MAX_ANNOTATIONS is not None:
    all_mat_files = all_mat_files[:MAX_ANNOTATIONS]

print(f"Found {len(all_mat_files)} annotation files to split across folds.")


# KFold split by annotation files
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_metrics = []
aggregate_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_mat_files), start=1):
    print("\n" + "=" * 60)
    print(f"FOLD {fold_idx}/{N_SPLITS}")
    print("=" * 60)

    train_files = [all_mat_files[i] for i in train_idx]
    val_files   = [all_mat_files[i] for i in val_idx]
    print(f"Train annotations: {len(train_files)} | Val annotations: {len(val_files)}")

    # Create datasets that load the listed mat files
    train_dataset = FusionDataset(
        folder=DATA_PATH,
        window_size=WINDOW_SIZE,
        interp_factor=INTERP_FACTOR,
        max_annotations=None,
        skip_first_n=0,
        train_mode=True,
        image_size=IMAGE_SIZE,
        local_prefix="C:/bachelorProject",
        cluster_prefix="/scratch/scratch-hdd/4all/mskrzypczyk",
        mat_files_list=train_files
    )

    val_dataset = FusionDataset(
        folder=DATA_PATH,
        window_size=WINDOW_SIZE,
        interp_factor=INTERP_FACTOR,
        max_annotations=None,
        skip_first_n=0,
        train_mode=False,
        image_size=IMAGE_SIZE,
        local_prefix="C:/bachelorProject",
        cluster_prefix="/scratch/scratch-hdd/4all/mskrzypczyk",
        mat_files_list=val_files
    )

    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples for fold {fold_idx}.")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model (fresh per fold)
    model = FusionNet(num_classes=NUM_CLASSES, mode=MODE, in_ch=1, feature_dim=FEATURE_DIM)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.8, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        # Train
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loop = tqdm(train_loader, desc="Train", ncols=100)
        for window, feats, img, label in loop:
            window = window.to(DEVICE)
            feats = feats.to(DEVICE)
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            if MODE == "fusion":
                outputs = model(frame=img, x_seq=window, x_feat=feats)
            elif MODE == "video":
                outputs = model(frame=img, x_seq=None, x_feat=None)
            elif MODE == "eog":
                outputs = model(frame=None, x_seq=window, x_feat=feats)

            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = outputs.argmax(dim=1)
            total_correct += (preds == label).sum().item()
            bs = label.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # Validation
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        loop = tqdm(val_loader, desc="Val", ncols=100)
        with torch.no_grad():
            for window, feats, img, label in loop:
                window = window.to(DEVICE)
                feats = feats.to(DEVICE)
                img = img.to(DEVICE)
                label = label.to(DEVICE)

                if MODE == "fusion":
                    outputs = model(frame=img, x_seq=window, x_feat=feats)
                elif MODE == "video":
                    outputs = model(frame=img, x_seq=None, x_feat=None)
                elif MODE == "eog":
                    outputs = model(frame=None, x_seq=window, x_feat=feats)

                loss = criterion(outputs, label)
                preds = outputs.argmax(dim=1)

                total_correct += (preds == label).sum().item()
                bs = label.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(label.cpu().numpy().tolist())

        val_loss = total_loss / total_samples
        val_acc  = total_correct / total_samples

        # Scheduler step
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"Learning Rate: {lr_now:.6f}")

        # Save best
        fold_dir = os.path.join(OUT_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(fold_dir, f"fold_{fold_idx}_best.pth")
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"New best model saved for fold {fold_idx} -> {save_path} (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{PATIENCE} epochs")
            if patience_counter >= PATIENCE:
                print("Early stopping for this fold.")
                break

    # Evaluate best model on validation set again and save confusion matrix + metrics
    print("\nEvaluating best checkpoint for this fold...")
    # load best weights
    best_ckpt = os.path.join(fold_dir, f"fold_{fold_idx}_best.pth")
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()

    # Run inference on validation loader to get predictions
    all_preds = []
    all_targets = []
    all_probs = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for window, feats, img, label in tqdm(val_loader, desc="Infer"):
            window = window.to(DEVICE)
            feats = feats.to(DEVICE)
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            if MODE == "fusion":
                logits = model(frame=img, x_seq=window, x_feat=feats)
            elif MODE == "video":
                logits = model(frame=img)
            elif MODE == "eog":
                logits = model(frame=None, x_seq=window, x_feat=feats)

            probs = softmax(logits).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(label.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds, dtype=int)
    all_targets = np.array(all_targets, dtype=int)
    all_probs = np.array(all_probs, dtype=float)

    # Metrics & confusion matrix
    if len(all_preds) == 0:
        print("!No predictions for this fold (empty val set?).")
        continue

    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(NUM_CLASSES))

    # Save metrics CSV for this fold
    metrics_path = os.path.join(fold_dir, f"fold_{fold_idx}_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "val_acc"])
        writer.writerow([fold_idx, acc])
        writer.writerow([])
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for cls in CLASS_NAMES:
            d = report[cls]
            writer.writerow([cls, d["precision"], d["recall"], d["f1-score"], int(d["support"])])
        macro = report["macro avg"]
        writer.writerow([])
        writer.writerow(["macro_avg", macro["precision"], macro["recall"], macro["f1-score"], ""])

    # Save per-sample CSV predictions for this fold
    pred_csv = os.path.join(fold_dir, f"fold_{fold_idx}_predictions.csv")
    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "true", "pred", "max_prob"])
        for i, (t, p, pv) in enumerate(zip(all_targets.tolist(), all_preds.tolist(), all_probs.tolist())):
            writer.writerow([i, CLASS_NAMES[t], CLASS_NAMES[p], float(np.max(pv))])

    # Save confusion matrix image
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im0 = ax[0].imshow(cm, interpolation="nearest", cmap="Blues")
    ax[0].set_title(f"Fold {fold_idx} Confusion (counts)")
    ax[0].set_xticks(np.arange(NUM_CLASSES)); ax[0].set_yticks(np.arange(NUM_CLASSES))
    ax[0].set_xticklabels(CLASS_NAMES, rotation=45); ax[0].set_yticklabels(CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[0].text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax[1].set_title(f"Fold {fold_idx} Confusion (row-normalized)")
    ax[1].set_xticks(np.arange(NUM_CLASSES)); ax[1].set_yticks(np.arange(NUM_CLASSES))
    ax[1].set_xticklabels(CLASS_NAMES, rotation=45); ax[1].set_yticklabels(CLASS_NAMES)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im1, ax=ax[1])
    plt.tight_layout()
    cm_path = os.path.join(fold_dir, f"fold_{fold_idx}_confusion.png")
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"✔ Saved fold {fold_idx} confusion -> {cm_path}")
    print(f"✔ Saved fold {fold_idx} metrics -> {metrics_path}")

    # accumulate for overall aggregated confusion matrix
    aggregate_cm += cm

    # collect fold summary
    fold_metrics.append({
        "fold": fold_idx,
        "val_acc": float(acc),
        "num_val_samples": int(len(all_targets))
    })


# Save aggregated confusion matrices + metrics CSV
if len(fold_metrics) == 0:
    print("No folds produced results.")
else:
    # Save fold summary CSV
    summary_csv = os.path.join(OUT_DIR, "kfold_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "val_acc", "num_val_samples"])
        for d in fold_metrics:
            writer.writerow([d["fold"], d["val_acc"], d["num_val_samples"]])
    print(f"KFold summary saved -> {summary_csv}")

    # Overall confusion matrix
    overall_cm = aggregate_cm
    overall_cm_norm = overall_cm.astype(float) / (overall_cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im0 = ax[0].imshow(overall_cm, interpolation="nearest", cmap="Blues")
    ax[0].set_title("Overall Confusion (sum of folds - counts)")
    ax[0].set_xticks(np.arange(NUM_CLASSES)); ax[0].set_yticks(np.arange(NUM_CLASSES))
    ax[0].set_xticklabels(CLASS_NAMES, rotation=45); ax[0].set_yticklabels(CLASS_NAMES)
    for i in range(overall_cm.shape[0]):
        for j in range(overall_cm.shape[1]):
            ax[0].text(j, i, int(overall_cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(overall_cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax[1].set_title("Overall Confusion (row-normalized average)")
    ax[1].set_xticks(np.arange(NUM_CLASSES)); ax[1].set_yticks(np.arange(NUM_CLASSES))
    ax[1].set_xticklabels(CLASS_NAMES, rotation=45); ax[1].set_yticklabels(CLASS_NAMES)
    for i in range(overall_cm_norm.shape[0]):
        for j in range(overall_cm_norm.shape[1]):
            ax[1].text(j, i, f"{overall_cm_norm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im1, ax=ax[1])
    plt.tight_layout()
    overall_cm_path = os.path.join(OUT_DIR, "kfold_overall_confusion.png")
    plt.savefig(overall_cm_path, dpi=150)
    plt.close(fig)
    print(f"Aggregate confusion matrices saved -> {overall_cm_path}")
