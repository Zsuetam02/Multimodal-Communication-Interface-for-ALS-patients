# Multimodal Communication Interface for ALS Patients

A biomedical engineering project focused on assistive communication for patients with amyotrophic lateral sclerosis (ALS), combining **camera-based eye-gaze tracking** and **electrooculography (EOG)** signals in a unified multimodal system.

The interface is designed to work with each modality independently, while achieving the best performance when both are used together.

---

## Project Overview

Patients with late-stage ALS often retain eye movement while losing most voluntary motor control, making eye-based interfaces one of the most viable communication channels.

This project implements a **multimodal human–computer interface** that:
- Interprets user intent from **video-based gaze estimation**
- Interprets eye movement from **EOG biosignals**
- Fuses both modalities using deep learning
- Enables text-based communication with optional text-to-speech output

---

## Key Features

- **Multimodal input**
  - Camera-based eye tracking (computer vision + CNN)
  - EOG signal processing (time-series ML)
- **Independent modality operation**
  - Camera-only
  - EOG-only
  - Fused multimodal mode
- **Five-class gaze classification**
  - Up, Down, Left, Right, Closed
- **Custom communication interface**
  - Gaze-controlled virtual keyboard
  - Text-to-speech support
- **Robust ML pipeline**
  - Signal preprocessing & augmentation
  - Cross-validation
  - Confusion matrix analysis

---

## System Architecture

### Modalities
- **Camera branch (VideoNet)**
  - Modified ResNet50 (grayscale input)
  - Eye-region preprocessing using MediaPipe
  - Outputs a 2048-D feature embedding

- **EOG branch (EOGNet)**
  - Bi-directional LSTM for temporal dynamics
  - MLP for engineered signal features
  - Processes horizontal and vertical EOG channels

- **FusionNet**
  - Concatenates embeddings from both modalities
  - Joint classifier trained end-to-end
  - Graceful degradation if one modality is unavailable

---

## Dataset

- **Participants**: 12 healthy subjects
- **Sampling rate**: 25 Hz (synchronized EOG + camera)
- **Classes**: Up, Down, Left, Right, Closed
- **Recording conditions**:
  - Different gaze speeds and angles
  - With and without glasses
  - Variable lighting conditions
- **Annotations**:
  - Time-aligned EOG signals and image sequences
  - Balanced class distribution

> Dataset is not publicly available due to ethical and consent constraints.

---

## Technologies Used

- **Python 3.9–3.10**
- **PyTorch**
- **OpenCV**
- **MediaPipe**
- **NumPy / SciPy**
- **scikit-learn**
- **MATLAB (signal preprocessing)**
- **BIOPAC MP36 (EOG acquisition)**

---

## Results

- Multimodal fusion outperformed single-modality models
- Improved robustness to:
  - Lighting changes
  - Partial occlusion
  - Signal noise
- Consistent performance across 10-fold cross-validation

Detailed metrics, learning curves, and confusion matrices are available in the thesis.

---

## Limitations

- Data collected on healthy participants only
- Limited dataset size
- Controlled acquisition environment
- Real-world ALS patient testing remains future work

---

## Future Work

- Clinical validation with ALS patients
- Real-time optimization for embedded systems
- Adaptive calibration per user
- Extended command vocabulary
- Wearable camera and electrode integration

---

## Author

**Mateusz Skrzypczyk**  
Biomedical Engineering  
Specialization: Information Systems in Medicine

Bachelor thesis supervised by  
**prof. dr hab. inż. Paweł Badura**  
Department of Medical Informatics and Artificial Intelligence

---

## License

This project is provided for academic and research purposes.

---

## Trained networks accessibility 

For anyone interested, please contact me on mat.sk@op.pl to receive trained models, as well as detailed results contained in my thesis



