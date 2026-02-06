# Default model files (change later if you want):
#   videonet_best.pth, eognet_best.pth, fusion_best.pth

import os
import time
import threading
from threading import Lock
from queue import Queue
from collections import deque

import tkinter as tk
import pyttsx3
import winsound

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import torch
import torch.nn.functional as F
from torchvision import transforms

from ModelConfiguratorCl import FusionNet


# Config
VIDEO_MODEL_PATH = "videonet_best.pth"
EOG_MODEL_PATH = "eognet_best.pth"
FUSION_MODEL_PATH = "fusion_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.8
MOVE_INTERVAL = 0.75

# Label maps
IDX_TO_LABEL = {0: "up", 1: "down", 2: "left", 3: "right", 4: "closed"}
LABEL_TO_IDX = {v: k for k, v in IDX_TO_LABEL.items()}


# Helpers (from DataLoader)
class CLAHEEqualization:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, pil_image):
        img = np.array(pil_image)
        eq = self.clahe.apply(img)
        return Image.fromarray(eq)

def butter_lowpass(signal, cutoff=30, fs=250, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[0]):
        filtered[ch] = filtfilt(b, a, signal[ch])
    return filtered

# EOG preprocessing as in FusionDataset
def compute_eog_window_and_feats(raw_signal, window_size=48, interp_factor=4):
    """
    raw_signal: numpy array shape (2, N) or list-like
    returns (window_norm (2, window_size), feats (14,))
    """
    sig = np.asarray(raw_signal, dtype=np.float32)
    if sig.ndim == 1:
        sig = np.stack([sig, sig], axis=0)
    if sig.shape[0] != 2:
        raise ValueError("raw_signal must have shape (2, N)")

    # lowpass
    sig_filt = butter_lowpass(sig, cutoff=30, fs=250)

    # interp x4
    orig_len = sig_filt.shape[1]
    new_len = orig_len * interp_factor
    interp_sig = np.zeros((2, new_len), dtype=np.float32)
    for ch in range(2):
        x = np.arange(orig_len)
        f_interp = interp1d(x, sig_filt[ch], kind='linear', fill_value="extrapolate")
        x_new = np.linspace(0, orig_len - 1, new_len)
        interp_sig[ch] = f_interp(x_new)

    # take last window_size samples
    if new_len < window_size:
        pad = window_size - new_len
        window = np.pad(interp_sig, ((0,0),(pad,0)), mode='edge')[:, -window_size:]
    else:
        window = interp_sig[:, -window_size:]

    # normalize per channel
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-6
    window_norm = (window - mean) / std

    feats = []
    for ch in range(2):
        ch_data = window[ch]
        feats.extend([
            ch_data.mean(), ch_data.std(), ch_data.max(), ch_data.min(),
            ch_data.max() - ch_data.min(), ch_data[-1] - ch_data[0],
            np.sum(ch_data ** 2)
        ])
    feats = np.array(feats, dtype=np.float32)
    return window_norm, feats


# UI: KeyboardApp

class KeyboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("On-Screen Keyboard")
        self.root.state('zoomed')
        self.root.resizable(False, False)
        self.command_queue = Queue()
        self.current_prediction = ("None", 0.0)


        self.pred_label = tk.Label(self.root, text="Prediction: None", font=("Arial", 18), fg="white", bg="blue")
        self.pred_label.place(relx=1.0, x=-10, y=10, anchor='ne')
        self.text_area = tk.Text(self.root, height=5, font=("Arial", 24), wrap="word")
        self.text_area.pack(fill=tk.X, padx=10, pady=10, anchor="nw")

        # Keys
        self.keys = [
            ['1','2','3','4','5','6','7','8','9','0','-'],
            ['Q','W','E','R','T','Y','U','I','O','P','+'],
            ['A','S','D','F','G','H','J','K','L',':','"'],
            ['Z','X','C','V','B','N','M',',','.','?','!'],
            ['%','Backspace','Clear All','Space','Submit TTS']
        ]

        # Keyboard frame anchored bottom-left
        self.keyboard_frame = tk.Frame(self.root)
        self.keyboard_frame.pack(side="bottom", anchor="s", padx=10, pady=10)

        # Create buttons
        self.buttons = []
        for r_idx, row in enumerate(self.keys):
            button_row = []
            col_counter = 0
            for key in row:
                display_key = "Submit\nTTS" if key == "Submit TTS" else key
                btn = tk.Button(self.keyboard_frame, text=display_key, font=("Arial", 18),
                                width=8 if key!='Space' else 24, height=3,
                                justify="center", command=lambda k=key: self.press_key(k))
                if key == 'Space':
                    btn.grid(row=r_idx, column=col_counter, columnspan=3, padx=2, pady=5, sticky='ew')
                    col_counter += 3
                else:
                    btn.grid(row=r_idx, column=col_counter, padx=2, pady=5)
                    col_counter += 1
                button_row.append(btn)
            self.buttons.append(button_row)

        # current selection
        self.current_row = 0
        self.current_col = 0
        self.highlight_selection()

        self.root.bind('<Up>', self.move_up)
        self.root.bind('<Down>', self.move_down)
        self.root.bind('<Left>', self.move_left)
        self.root.bind('<Right>', self.move_right)
        self.root.bind('<Return>', self.select_key)

        self.root.after(50, self.check_queue)

    def highlight_selection(self):
        for row in self.buttons:
            for btn in row:
                btn.config(bg='SystemButtonFace')
        # bounds-check current indices
        if self.current_row >= len(self.buttons): self.current_row = 0
        if self.current_col >= len(self.buttons[self.current_row]): self.current_col = 0
        self.buttons[self.current_row][self.current_col].config(bg='lightblue')
        winsound.Beep(1000, 50)

    def move_up(self, event=None):
        self.current_row = (self.current_row - 1) % len(self.buttons)
        self.adjust_column()
        self.highlight_selection()

    def move_down(self, event=None):
        self.current_row = (self.current_row + 1) % len(self.buttons)
        self.adjust_column()
        self.highlight_selection()

    def move_left(self, event=None):
        self.current_col = (self.current_col - 1) % len(self.buttons[self.current_row])
        self.highlight_selection()

    def move_right(self, event=None):
        self.current_col = (self.current_col + 1) % len(self.buttons[self.current_row])
        self.highlight_selection()

    def adjust_column(self):
        if self.current_col >= len(self.buttons[self.current_row]):
            self.current_col = len(self.buttons[self.current_row]) - 1

    def select_key(self, event=None):
        key = self.keys[self.current_row][self.current_col]
        self.press_key(key)

    def press_key(self, key):
        if key == 'Backspace':
            self.text_area.delete('end-2c', 'end-1c')
        elif key == 'Clear All':
            self.text_area.delete(1.0, tk.END)
        elif key == 'Space':
            self.text_area.insert(tk.END, ' ')
        elif key == 'Submit TTS':
            self.submit_tts()
        else:
            self.text_area.insert(tk.END, key)
        winsound.Beep(1500, 50)

    def submit_tts(self):
        text = self.text_area.get(1.0, tk.END).strip()
        if text:
            threading.Thread(target=self._speak_text, args=(text,), daemon=True).start()

    def _speak_text(self, text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def process_external_input(self, command, confidence):
        self.current_prediction = (command, confidence)
        self.pred_label.config(text=f"Prediction: {command} ({confidence:.2f})")
        if confidence >= THRESHOLD:
            if command == "up":
                self.move_up()
            elif command == "down":
                self.move_down()
            elif command == "left":
                self.move_left()
            elif command == "right":
                self.move_right()
            elif command in ("select", "closed"):
                self.select_key()

    def check_queue(self):
        while not self.command_queue.empty():
            cmd, conf = self.command_queue.get()
            self.process_external_input(cmd, conf)
        self.root.after(50, self.check_queue)

# ====================================================
# Video thread
# - in standalone video mode it loads FusionNet(mode="video") and runs full classification
# - in fusion mode emit_for_fusion=True -> it DOES NOT load a classifier; it only sends preprocessed frame tensors to fusion manager
# ====================================================
class EyeTrackerThread(threading.Thread):
    def __init__(self, queue, model_path=VIDEO_MODEL_PATH, emit_for_fusion=False, fusion_queue=None):
        super().__init__(daemon=True)
        self.queue = queue
        self.emit_for_fusion = emit_for_fusion
        self.fusion_queue = fusion_queue
        self.cap = cv2.VideoCapture(0)
        self.model = None

        # Only load the classifier model when NOT in fusion emission mode.
        if not self.emit_for_fusion:
            try:
                self.model = FusionNet(num_classes=5, mode="video", in_ch=1).to(DEVICE)
                ckpt = torch.load(model_path, map_location=DEVICE)
                if list(ckpt.keys())[0].startswith("module."):
                    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
                self.model.load_state_dict(ckpt)
                self.model.eval()
                print(f"✅ Loaded video model from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load video model from {model_path}: {e}. Video predictions disabled.")
                self.model = None
        else:
            # Fusion emission mode: do not load any video classifier weights.
            print("Video thread running in embedding-only mode (no classifier loaded).")

        # face mesh & preprocessing
        import mediapipe as mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pred_buffer = deque(maxlen=7)
        self.smooth_width = None
        self.alpha = 0.15
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.bias = torch.tensor([0.0, 2.5, 0.0, -3.5, 2.5], device=DEVICE)
        self.last_move_time = 0.0

    def smooth_prediction(self, probs):
        self.pred_buffer.append(probs.cpu().numpy())
        if len(self.pred_buffer) < 5:
            return probs, False
        avg_probs = torch.tensor(np.mean(self.pred_buffer, axis=0), device=DEVICE)
        return avg_probs, True

    def capture_roi(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        h,w,_ = frame.shape
        lx,ly = lm[468].x*w, lm[468].y*h
        rx,ry = lm[473].x*w, lm[473].y*h
        pupil_dist = np.hypot(lx-rx, ly-ry)
        if pupil_dist < 5:
            return None

        TARGET_PUPIL_RESIZED = 146
        RESIZED_WIDTH = 256
        desired_width = (pupil_dist*RESIZED_WIDTH)/TARGET_PUPIL_RESIZED
        if self.smooth_width is None:
            self.smooth_width = desired_width
        else:
            self.smooth_width = self.alpha*desired_width + (1-self.alpha)*self.smooth_width
        roi_w = int(self.smooth_width)
        roi_h = max(8, roi_w // 4)
        cx = int((lx+rx)/2)
        cy = int((ly+ry)/2)
        xMin = max(0, cx - roi_w//2)
        xMax = min(w, cx + roi_w//2)
        yMin = max(0, cy - roi_h//2)
        yMax = min(h, cy + roi_h//2)
        if xMax <= xMin or yMax <= yMin:
            return None
        roi = frame[yMin:yMax, xMin:xMax]
        return roi

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            roi = self.capture_roi(frame)
            if roi is not None:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_3d = np.expand_dims(gray, axis=2)
                roi_tensor = self.preprocess(gray_3d).unsqueeze(0)  # (1,1,256,64)
                if self.emit_for_fusion:
                    # Fusion mode: don't run classifier, just send cpu tensors to fusion manager
                    if self.fusion_queue is not None:
                        try:
                            self.fusion_queue.put(("video_embed", roi_tensor.cpu()))
                        except Exception as e:
                            print("Failed to put video embed to fusion queue:", e)
                else:
                    # Standalone video classification
                    if self.model is not None:
                        try:
                            roi_tensor = roi_tensor.to(DEVICE)
                            with torch.no_grad():
                                logits = self.model(frame=roi_tensor)
                                probs = F.softmax(logits.squeeze(), dim=0)
                                smooth_probs, stable = self.smooth_prediction(probs)
                                if stable:
                                    conf, pred_idx = torch.max(smooth_probs, 0)
                                    label = IDX_TO_LABEL[int(pred_idx.item())]
                                    conf_val = float(conf.item())
                                    # throttle moves
                                    if time.time() - self.last_move_time >= MOVE_INTERVAL:
                                        if self.queue is not None:
                                            self.queue.put((label, conf_val))
                                        self.last_move_time = time.time()
                        except Exception as e:
                            print("Video inference error:", e)
                    else:
                        # model not loaded: do nothing
                        pass
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()


# EOG-only handler (loads EOG classifier) - used in EOG-only mode
class EOGHandlerClassifier:
    def __init__(self, app_queue: Queue, model_path=EOG_MODEL_PATH, window_size=48, interp_factor=4):
        self.app_queue = app_queue
        self.lock = Lock()
        self.pending = []
        self.window_size = window_size
        self.interp_factor = interp_factor
        self._running = True

        # Load EOG classifier (FusionNet in eog mode)
        self.model = None
        try:
            self.model = FusionNet(num_classes=5, mode="eog", in_ch=1).to(DEVICE)
            ckpt = torch.load(model_path, map_location=DEVICE)
            if list(ckpt.keys())[0].startswith("module."):
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            self.model.load_state_dict(ckpt)
            self.model.eval()
            print(f"✅ Loaded EOG model from {model_path}")
            self.model_loaded = True
        except Exception as e:
            print(f"⚠️ Could not load EOG model from {model_path}: {e}. EOG predictions disabled.")
            self.model = None
            self.model_loaded = False

        # worker thread
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def Receive_external_signal(self, raw_signal):
        """
        PUBLIC API: push raw EOG here. raw_signal should be numpy array shape (2, N)
        Not yet implemented
        """
        with self.lock:
            self.pending.append(np.asarray(raw_signal, dtype=np.float32))

    def _worker_loop(self):
        while self._running:
            item = None
            with self.lock:
                if self.pending:
                    item = self.pending.pop(0)
            if item is None:
                time.sleep(0.01)
                continue

            try:
                window_norm, feats = compute_eog_window_and_feats(item, window_size=self.window_size, interp_factor=self.interp_factor)
            except Exception as e:
                print("EOG preprocessing error:", e)
                continue

            if self.model_loaded and self.model is not None:
                try:
                    x_seq = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,2,48)
                    x_feat = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)      # (1,14)
                    with torch.no_grad():
                        logits = self.model(x_seq=x_seq, x_feat=x_feat)
                        probs = torch.softmax(logits.squeeze(), dim=0).cpu()
                        conf, pred_idx = torch.max(probs, 0)
                        label = IDX_TO_LABEL[int(pred_idx.item())]
                        self.app_queue.put((label, float(conf.item())))
                except Exception as e:
                    print("EOG inference error:", e)
            else:
                # model not loaded / disabled
                pass

    def stop(self):
        self._running = False
        self.worker.join(timeout=1.0)


# EOG fusion handler (no classifier loaded) - used in fusion mode
class EOGHandlerFusion:
    def __init__(self, fusion_queue: Queue, window_size=48, interp_factor=4):
        self.fusion_queue = fusion_queue
        self.lock = Lock()
        self.pending = []
        self.window_size = window_size
        self.interp_factor = interp_factor
        self._running = True

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def Receive_external_signal(self, raw_signal):
        with self.lock:
            self.pending.append(np.asarray(raw_signal, dtype=np.float32))

    def _worker_loop(self):
        while self._running:
            item = None
            with self.lock:
                if self.pending:
                    item = self.pending.pop(0)
            if item is None:
                time.sleep(0.01)
                continue

            try:
                window_norm, feats = compute_eog_window_and_feats(item, window_size=self.window_size, interp_factor=self.interp_factor)
            except Exception as e:
                print("EOG preprocessing error:", e)
                continue

            # push CPU tensors to fusion queue
            x_seq_cpu = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).cpu()
            feats_cpu = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).cpu()
            try:
                self.fusion_queue.put(("eog_window", x_seq_cpu, feats_cpu))
            except Exception as e:
                print("Failed to put eog_window into fusion queue:", e)

    def stop(self):
        self._running = False
        self.worker.join(timeout=1.0)


# Fusion manager — waits for video frame tensors and eog_window and runs FusionNet(mode="fusion")
class FusionManager:
    def __init__(self, app_queue: Queue, fusion_model_path=FUSION_MODEL_PATH):
        self.shared_queue = Queue()
        self.app_queue = app_queue
        self.lock = Lock()
        self.latest_video = None
        self.latest_eog = None
        self._running = True

        self.model = None
        try:
            self.model = FusionNet(num_classes=5, mode="fusion", in_ch=1).to(DEVICE)
            ckpt = torch.load(fusion_model_path, map_location=DEVICE)
            if list(ckpt.keys())[0].startswith("module."):
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            self.model.load_state_dict(ckpt)
            self.model.eval()
            self.model_loaded = True
            print(f"✅ Loaded fusion model from {fusion_model_path}")
        except Exception as e:
            print(f"⚠️ Could not load fusion model from {fusion_model_path}: {e}. Fusion disabled.")
            self.model = None
            self.model_loaded = False

        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def put(self, item):
        self.shared_queue.put(item)

    def _loop(self):
        while self._running:
            try:
                item = self.shared_queue.get(timeout=0.1)
            except:
                item = None
            if item is None:
                continue
            src = item[0]
            if src == "video_embed":
                with self.lock:
                    # item: ("video_embed", roi_tensor_cpu)
                    self.latest_video = item
            elif src == "eog_window":
                with self.lock:
                    # item: ("eog_window", x_seq_cpu, feats_cpu)
                    self.latest_eog = item

            with self.lock:
                if self.latest_video is not None and self.latest_eog is not None:
                    if self.model_loaded and self.model is not None:
                        try:
                            roi_tensor = self.latest_video[1].to(DEVICE)  # (1,1,256,64)
                            x_seq = self.latest_eog[1].to(DEVICE)        # (1,2,48)
                            x_feat = self.latest_eog[2].to(DEVICE)       # (1,14)
                            with torch.no_grad():
                                logits = self.model(frame=roi_tensor, x_seq=x_seq, x_feat=x_feat)
                                probs = torch.softmax(logits.squeeze(), dim=0)
                                conf, pred_idx = torch.max(probs, 0)
                                label = IDX_TO_LABEL[int(pred_idx.item())]
                                self.app_queue.put((label, float(conf.item())))
                        except Exception as e:
                            print("Fusion inference error:", e)
                    # clear latests to wait for next pair
                    self.latest_video = None
                    self.latest_eog = None

    def stop(self):
        self._running = False
        self.worker.join(timeout=1.0)


# Top-level start function that wires everything depending on SELECTED_MODALITY
def start_pipeline_for_modality(modality_str, app_queue: Queue):
    """
    modality_str: "video", "eog", "video_eog"
    returns a dict of handlers
    """
    handlers = {}

    if modality_str == "video":
        # start video-only (full classification)
        video_thread = EyeTrackerThread(app_queue, model_path=VIDEO_MODEL_PATH, emit_for_fusion=False, fusion_queue=None)
        video_thread.start()
        handlers["video_thread"] = video_thread
        return handlers

    elif modality_str == "eog":
        # EOG-only: classifier version
        eog_handler = EOGHandlerClassifier(app_queue, model_path=EOG_MODEL_PATH, window_size=48, interp_factor=4)
        handlers["eog_handler"] = eog_handler
        return handlers

    elif modality_str in ("video_eog", "fusion"):
        # fusion: create fusion manager, start video that emits frames to fusion_manager, and EOG fusion handler
        fusion_manager = FusionManager(app_queue, fusion_model_path=FUSION_MODEL_PATH)
        handlers["fusion_manager"] = fusion_manager

        # video: embedding-only; do NOT load video classifier
        video_thread = EyeTrackerThread(queue=None, model_path=VIDEO_MODEL_PATH, emit_for_fusion=True, fusion_queue=fusion_manager.shared_queue)
        video_thread.start()
        handlers["video_thread"] = video_thread

        # EOG: fusion-only handler (no EOG classifier loaded)
        fusion_eog = EOGHandlerFusion(fusion_queue=fusion_manager.shared_queue, window_size=48, interp_factor=4)
        handlers["eog_handler"] = fusion_eog

        return handlers

    else:
        raise ValueError(f"Unknown modality: {modality_str}")

# Modality selection popup
SELECTED_MODALITY = None

def choose_modality():
    global SELECTED_MODALITY
    popup = tk.Tk()
    popup.title("Choose Modality")
    popup.geometry("320x180")
    popup.resizable(False, False)

    label = tk.Label(popup, text="Choose modality:", font=("Arial", 14))
    label.pack(pady=12)

    def set_modality(value):
        global SELECTED_MODALITY
        SELECTED_MODALITY = value
        popup.destroy()

    btn_video = tk.Button(popup, text="Video", font=("Arial", 12), width=20, command=lambda: set_modality("video"))
    btn_eog = tk.Button(popup, text="EOG", font=("Arial", 12), width=20, command=lambda: set_modality("eog"))
    btn_both = tk.Button(popup, text="Video + EOG", font=("Arial", 12), width=20, command=lambda: set_modality("video_eog"))

    btn_video.pack(pady=4)
    btn_eog.pack(pady=4)
    btn_both.pack(pady=4)

    popup.mainloop()

# MAIN
if __name__ == "__main__":
    choose_modality()
    print("Selected modality:", SELECTED_MODALITY)

    root = tk.Tk()
    app = KeyboardApp(root)

    handlers = start_pipeline_for_modality(SELECTED_MODALITY, app.command_queue)

    # raw_signal_array should be numpy array shape (2, N)
    #
    # Note:
    # - In video_only mode handlers["video_thread"] is started and will push (label,conf) to app.command_queue.
    # - In eog_only mode handlers["eog_handler"] (classifier) will push (label,conf) to app.command_queue.
    # - In fusion mode handlers["fusion_manager"] will push (label,conf) to app.command_queue after it has both a frame and an EOG window.

    root.mainloop()

    # On exit, stop background threads gracefully if present
    try:
        if "eog_handler" in handlers and hasattr(handlers["eog_handler"], "stop"):
            handlers["eog_handler"].stop()
        if "fusion_manager" in handlers and hasattr(handlers["fusion_manager"], "stop"):
            handlers["fusion_manager"].stop()
        if "video_thread" in handlers and hasattr(handlers["video_thread"], "join"):
            # Camera thread will stop when mainloop ends because it checks camera
            pass
    except Exception:
        pass
