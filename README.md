# Detecting Concept Shifts under Different Levels of Self-awareness on Labeling


## Dataset 
- **SEED** : 3 class *(netral, positive, negative)*
- **SEED-IV** : 4 class *(happiness, sadness, fear, neutral)*
- **GAMEEMO** : 9 class  *(valence, arousal)*
---

### **Preprocessing** (Offline Transform, Feature Extraction ...)
**0.0 preprocessing.py**<br>
*Temporal Segmentation (Sliding Window, Time Partitioning, Time Window, ...)*<br>
\+ *Feature Extraction (Signal Transformation, Dimensionality Reduction, ...)*
- Segmentation (Raw Signal)
- Segmentation + DE (Differential Entropy)
- Segmentation + PSD (Power Spectral Density)
    | | Seg<br>`(channels, window)` | Seg + DE<br>`(channels, 4 bands)` | Seg + PSD<br>`(channels, 4 bands)` |
    | --- | :---:| :---: | :---: |
    | **SEED**    | `(62, 400)` | `(62, 4)` | `(62, 4)` |
    | **SEED-IV** | `(62, 400)` | `(62, 4)` | `(62, 4)` |
    | **GAMEEMO** | `(14, 256)` | `(14, 4)` | `(14, 4)` |

`EEG channels(num_electrodes), Segment size(Window size)`

### **Make Dataset** 
train and test data are split for reliable generalization evaluation.<br>
**0.1 make dataset.py** : *( train : test = 9 : 1 )*

---

## Our Method

### **1) Subject-dependent Train, Test**  
**0.2 subdepend.py**: For each subject, **subdepend.py** is executed.

Example.
```
python ./0.2\ subdepend.py --dataset=GAMEEMO --label=v --model=CCNN --feature=DE --epoch=100
```

### **2) Make OOD Detection Model**  
**1.0 OOD detector.py**<br>
- Data Pipeline
  - scaling
    - Raw Signal - standardization
    - PSD - log transformation
  - deshape(reshape)
    - CCNN - grid: make 9x9 grid (samples, channels, 4 bands) -> (samples, 4 bands, 9, 9)
    - TSC - expand : (samples, channels, window) -> (samples, 1, channels, window)
  
- Model 
  - CCNN : https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
  - TSCeption : https://arxiv.org/abs/2104.02935
  - DGCNN : https://ieeexplore.ieee.org/abstract/document/8320798
  - All models are based on TorchEEG : https://github.com/torcheeg/torcheeg
- criterion = `torch.nn.CrossEntropyLoss()`
- optimizer = `torch.optim.Adam(model.parameters())`
- scheduler = `CosineAnnealingWarmUpRestarts(optimizer)`
- Please note the paper to take detailed paremeters.


Example.
```
python ./1.0\ OOD\ detector.py --dataset=GAMEEMO --label=v --model=CCNN --feature=DE --epoch=100 --cut=6
```

### **3) Remove OOD and Train ID**  
**2.0 Base_Remove.py**<br>
Set the threshold of ODM to adjust the OOD removal rate (approximately 10%).
Please note the paper to take detailed thresholds.


Example.
```
python ./2.0\ Base_Remove.py --dataset=GAMEEMO --label=v --model=CCNN --feature=DE --epoch=100 --detector=Low_6 --threshold=0.95
```

### **4) Compare with the Baseline**
In **2.0 Base_Remove.py**, `threshold==0` is equivalent to Baseline.

## LICENSE
This repository has a MIT license, as found in the [LICENSE](./LICENSE) file.