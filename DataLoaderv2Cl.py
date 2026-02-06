import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from PIL import Image
from torchvision import transforms
import cv2
import random
from torchvision.transforms import functional as F

# ----------------------------- LABEL MAP -----------------------------
LABEL_MAP = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "closed": 4
}

def h5_char_to_str(obj):
    data = obj[()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    if isinstance(data, str):
        return data
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.integer):
            return "".join(chr(c) for c in data.flatten() if c != 0)
        elif np.issubdtype(data.dtype, np.str_):
            return "".join(data.flatten())
    return str(data)

# ----------------------------- CUSTOM TRANSFORMS -----------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

class RandomZoom(object):
    def __init__(self, zoom_range=(0.9, 1.1), p=0.5):
        self.zoom_range = zoom_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        zoom_factor = random.uniform(*self.zoom_range)
        w, h = img.size

        if zoom_factor < 1.0:
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            img = F.crop(img, top, left, new_h, new_w)
            img = F.resize(img, (h, w), interpolation=transforms.InterpolationMode.BICUBIC)
        else:
            pad_w = int((w * zoom_factor - w) / 2)
            pad_h = int((h * zoom_factor - h) / 2)
            img = F.pad(img, (pad_w, pad_h, pad_w, pad_h), padding_mode='edge')
            img = F.resize(img, (h, w), interpolation=transforms.InterpolationMode.BICUBIC)

        return img

class RandomYRotation:
    def __init__(self, max_angle=30, p=0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        h, w = img.shape[:2]
        angle = random.uniform(-self.max_angle, self.max_angle)
        focal = w
        a = np.deg2rad(angle)
        R = np.array([[np.cos(a), 0, np.sin(a)],
                      [0, 1, 0],
                      [-np.sin(a), 0, np.cos(a)]])
        corners = np.array([[-w/2, -h/2, 0],
                            [ w/2, -h/2, 0],
                            [ w/2,  h/2, 0],
                            [-w/2,  h/2, 0]])
        rotated = corners @ R.T
        projected = rotated.copy()
        projected[:, 2] += focal
        projected = projected[:, :2] / projected[:, 2, np.newaxis] * focal
        projected += np.array([w/2, h/2])
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst = np.float32(projected)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped)

class RandomXRotation:
    def __init__(self, max_angle=30, p=0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        h, w = img.shape[:2]
        angle = random.uniform(-self.max_angle, self.max_angle)
        focal = h
        a = np.deg2rad(angle)
        R = np.array([[1, 0, 0],
                      [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a),  np.cos(a)]])
        corners = np.array([[-w/2, -h/2, 0],
                            [ w/2, -h/2, 0],
                            [ w/2,  h/2, 0],
                            [-w/2,  h/2, 0]])
        rotated = corners @ R.T
        projected = rotated.copy()
        projected[:, 2] += focal
        projected = projected[:, :2] / projected[:, 2, np.newaxis] * focal
        projected += np.array([w/2, h/2])
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32(projected)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped)

class CLAHEEqualization:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, pil_image):
        img = np.array(pil_image)
        eq = self.clahe.apply(img)
        return Image.fromarray(eq)

# ----------------------------- EOG PREPROCESSING -----------------------------
def butter_lowpass(signal, cutoff=30, fs=250, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[0]):
        filtered[ch] = filtfilt(b, a, signal[ch])
    return filtered

def random_amplitude_scaling(signal, scale_range=(0.9, 1.1), p=0.3):
    if random.random() < p:
        scale = random.uniform(*scale_range)
        signal = signal * scale
    return signal

# ----------------------------- FUSION DATASET -----------------------------
class FusionDataset(Dataset):

    def __init__(self,
                 folder,
                 window_size=48,
                 interp_factor=4,
                 max_annotations=900,
                 skip_first_n=0,
                 max_windows_per_signal=99999,
                 train_mode=True,
                 image_size=(256, 64),
                 local_prefix="C:/bachelorProject",
                 cluster_prefix="/scratch/scratch-hdd/4all/mskrzypczyk",
                 mat_files_list=None):

        self.folder = folder.replace("\\", "/")
        self.window_size = window_size
        self.interp_factor = interp_factor
        self.samples = []
        self.train_mode = train_mode
        self.image_size = image_size
        self.local_prefix = local_prefix.replace("\\", "/")
        self.cluster_prefix = cluster_prefix.replace("\\", "/")

        # -----------------------------
        # UPDATED MAT FILE DISCOVERY
        # -----------------------------
        if mat_files_list is None:
            # Default: load all .mat files in folder
            mat_files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])
        else:
            # K-fold mode: use the provided subset
            available = set(os.listdir(folder))
            # Keep only filenames that actually exist in the directory
            mat_files = [f for f in mat_files_list if f in available]

        if skip_first_n > 0:
            mat_files = mat_files[skip_first_n:]

        if max_annotations is not None:
            mat_files = mat_files[:max_annotations]

        print(f"➡️ Using {len(mat_files)} annotation files out of {len(os.listdir(folder))}")


        processed_annotations = 0
        skipped_annotations = 0
        skipped_due_to_window = 0

        for file in mat_files:
            filepath = os.path.join(folder, file)
            print(f"\n🔹 Processing annotation: {file}")
            with h5py.File(filepath, "r") as f:
                label_str = h5_char_to_str(f["entry"]["Label"])
                print(f"Label string: {label_str}")
                label = LABEL_MAP[label_str.lower()]

                img_folder_path = h5_char_to_str(f["entry"]["Image"])
                img_folder_path = self._fix_path(img_folder_path)
                print(f"Resolved image folder: {img_folder_path}")

                if "small" in img_folder_path.lower():
                    print("Skipped (folder contains 'small')")
                    skipped_annotations += 1
                    continue
                if not os.path.isdir(img_folder_path):
                    print("Skipped (folder missing)")
                    skipped_annotations += 1
                    continue

                img_paths = sorted([
                    os.path.join(img_folder_path, x).replace("\\", "/")
                    for x in os.listdir(img_folder_path)
                    if x.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                print(f"Found {len(img_paths)} images in folder")
                if len(img_paths) == 0:
                    print("Skipped (no images in folder)")
                    skipped_annotations += 1
                    continue

                two_thirds = len(img_paths) * 2 // 3
                used_imgs = img_paths[-two_thirds:]
                print(f"Using last 2/3 images: {len(used_imgs)} frames")

                sig1 = f["entry"]["Signal1"][()].astype(np.float32).flatten()
                sig2 = f["entry"]["Signal2"][()].astype(np.float32).flatten()
                signal = np.stack([sig1, sig2], axis=0)

                # Apply low-pass filter
                signal = butter_lowpass(signal, cutoff=30, fs=250)

                orig_len = signal.shape[1]
                new_len = orig_len * self.interp_factor
                interp_sig = np.zeros((2, new_len), dtype=np.float32)
                for ch in range(2):
                    x = np.arange(orig_len)
                    f_interp = interp1d(x, signal[ch], kind='linear')
                    x_new = np.linspace(0, orig_len - 1, new_len)
                    interp_sig[ch] = f_interp(x_new)
                print(f"Interpolated signal shape: {interp_sig.shape}")

                total_windows = interp_sig.shape[1] - self.window_size + 1
                print(f"Total sliding windows possible (interp signal): {total_windows}")

                center_offset = (self.window_size // 2) - 1
                for start in range(total_windows):
                    center_idx = start + center_offset

                    # Map center index from interpolated signal to image index
                    center_idx_img = int(center_idx / self.interp_factor)
                    if center_idx_img < 0 or center_idx_img >= len(used_imgs):
                        # skip windows that map outside image range
                        continue

                    end = start + self.window_size
                    window = interp_sig[:, start:end]

                    # Random amplitude scaling augmentation
                    if self.train_mode:
                        window = random_amplitude_scaling(window, scale_range=(0.9, 1.1), p=0.3)

                    # Normalize window
                    mean = window.mean(axis=1, keepdims=True)
                    std = window.std(axis=1, keepdims=True) + 1e-6
                    window_norm = (window - mean) / std

                    # Compute simple features
                    feats = []
                    for ch in range(2):
                        ch_data = window[ch]
                        feats.extend([
                            ch_data.mean(),
                            ch_data.std(),
                            ch_data.max(),
                            ch_data.min(),
                            ch_data.max() - ch_data.min(),
                            ch_data[-1] - ch_data[0],
                            np.sum(ch_data ** 2)
                        ])
                    feats = np.array(feats, dtype=np.float32)

                    # Save sample
                    self.samples.append((window_norm, feats, used_imgs[center_idx_img], label))

                processed_annotations += 1

        print(f"\nFusion dataset created with {len(self.samples)} samples")
        print(f"Processed annotations: {processed_annotations}")
        print(f"Skipped annotations (folder issues): {skipped_annotations}")
        print(f"Skipped due to insufficient windows: {skipped_due_to_window}")

        # ----------------------------- IMAGE TRANSFORMS -----------------------------
        if train_mode:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                CLAHEEqualization(),
                RandomZoom(zoom_range=(0.8, 1.2), p=0.6),
                RandomYRotation(max_angle=35, p=0.7),
                RandomXRotation(max_angle=35, p=0.7),
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                AddGaussianNoise(mean=0.0, std=0.02, p=0.3),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                CLAHEEqualization(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def _fix_path(self, path):
        path = path.replace("\\", "/")
        if "ExportAppViewer" in path:
            idx = path.index("ExportAppViewer")
            path = os.path.join(self.cluster_prefix, path[idx:]).replace("\\", "/")
        else:
            path = path.replace(self.local_prefix, self.cluster_prefix)
        return path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, feats, img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        return (
            torch.from_numpy(window).float(),
            torch.from_numpy(feats).float(),
            img.float(),
            torch.tensor(label, dtype=torch.long)
        )
