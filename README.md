# Persistent-Homology-vs.-Learning-Methods
Official implementation for STAG 2024 paper

![schema_svg](https://github.com/user-attachments/assets/658818f9-f187-42f1-bdaf-6469a48a6e83)

# Abstract
This exploratory study compares persistent homology methods with traditional machine learning and deep learning techniques
for label-efficient classification. We propose pure topological approaches, including persistence thresholding and Bottleneck
distance classification, and explore hybrid methods combining persistent homology with machine learning. These are evaluated
against conventional machine learning algorithms and deep neural networks on two binary classification tasks: surface crack
detection and malaria cell identification. We assess performance across sample sizes per class, ranging from 1 to 500.
Our study highlights the effectiveness of persistent homology-based methods in low-data scenarios. Using the Bottleneck dis-
tance approach, we achieve 95.95% accuracy in crack detection and 93.11% in malaria diagnosis with only one labeled sample
per class. These results outperform the best performance from machine learning models, which achieves 69.40% and 39.75%
accuracy, respectively, and deep learning models, which attains up to 95.96% in crack detection and 62.72% in malaria diag-
nosis. This demonstrates the superior performance of topological methods in classification tasks with few labeled data.
Hybrid approaches demonstrate enhanced performance as sample sizes increase, effectively leveraging topological features
to boost classification accuracy. This study highlights the robustness of topological methods in extracting meaningful features
from limited data, offering promising directions for efficient, label-conserving classification strategies. The results underscore
the worth of persistent homology, both as a standalone tool and in combination with machine learning, particularly in domains
where labeled data scarcity challenges traditional deep learning approaches.

# Getting Started
## Installation
Install python packages
```
pip install -r requirements.txt
```

## Preparing Datasets
Download the [cracks](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) dataset and prepare the following directory structure.
As for the malaria dataset, there's no need to download the images manually, as they are automatically sourced from the TensorFlow library.

- datasets
    - cracks
        -  Negative
            - Negative_00001.jpg
            ...
            - Negative_20000.jpg
        -  Positive
            - Positive_00001.jpg
            ...
            - Positive_20000.jpg

         

## Running Experiments
To run the experiments, follow these steps:

### Persistent homology and Machine learning:
  ```
  python main_PH_ML.py

  ```

### Deep Learning:
  ```
  python main_DL.py

  ```




