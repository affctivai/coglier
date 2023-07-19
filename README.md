# Granularity
OOD for EEG signals


## Dataset 
- **DEAP** :  9 class  *(valence, arousal)*
- **SEED** : 
- **SEED IV** :
- **GAMEEMO** : 9 class  *(valence, arousal)*

---

### **preprocessing**
**0.0 preprocessing.py** : *Temporal Segmentation (Sliding Window, Time Partitioning, Time Window, ...)*
- No segmentation
- segmentation (raw)
- segmentation + DE
- segmentation + PSD

data shape: `(Samples, EEG channels(num_electrodes), Segment size(Window size))`

### **make dataset** 
**0.1 make dataset.py** : *( train : valid : test = 80 : 10 :10 )*

Subject-Independent / Subject-dependent

- deshape
  - 'gird' : make 9x9 grid `(samples, channels, 4 bands) -> (samples, 4 bands, 9, 9)`
  - 'expand' : `(samples, channels, window) -> (samples, 1, channels, window)`

- scaling
  - 'standard' : `sklearn.preprocessing.StandardScaler()`
  - 'log' : `numpy.log1p(x)`


### **baseline**  
**1.0 baseline.py** : *Subject-Independet train, test*

- model 
  - CCNN : https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
  - TSCeption : https://arxiv.org/abs/2104.02935
  - EEGNet : https://arxiv.org/abs/1611.08024
  - DGCNN : https://ieeexplore.ieee.org/abstract/document/8320798

- hyperparameter
    |        | CCNN   | TSCeption| EEGNet | DGCNN |
    | ---    | :----: | :-------:| :----: | :----: |
    | epoch  | 100    |  200     | 200    |      |
    | max_lr | 1e-4   | 1e-3     | 1e-3   |      |


- criterion = `torch.nn.CrossEntropyLoss()`
- optimizer = `torch.optim.Adam(model.parameters(), lr=0, weight_decay=1e-4)`
- scheduler = `CosineAnnealingWarmUpRestarts(optimizer, T_0=STEPS, T_mult=1, eta_max=max_lr, T_up=STEP*3, gamma=0.5)` 
  -  utils > scheduler.py



