| Model | Input | Seeds | Test accuracy | Test macro-F1 | Parameters | File (MB) | CPU latency (ms/image) | Training (min) |
|---|---|---|---|---|---|---|---|---|
| mobilenet_frozen_cosine | 64×64 | 3 | 46.5 ± 0.3 | 36.6 ± 0.1 | 2,278,955 | 9.34 | 5.1 | 8 |
| mobilenet_finetune | 64×64 | 3 | 86.6 ± 0.3 | 81.1 ± 0.9 | 2,278,955 | 9.34 | 4.8 | 23 |
| cnn_scratch | 64×64 | 3 | 95.3 ± 0.4 | 92.1 ± 1.1 | 75,099 | 0.31 | 2.0 | 20 |
| hog_svm | 64×64 | 1 | 91.2 ± 0.0 | 90.1 ± 0.0 | 75,895 | 0.61 | 1.0 | 5 |
