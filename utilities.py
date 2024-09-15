#----------------------------------------------------------------------------------------
#-------------------------------------------PACKAGES-------------------------------------
#----------------------------------------------------------------------------------------

import numpy as np
import os
import random
import tensorflow as tf
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import subprocess
import sys
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import importlib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from ripser import ripser

#----------------------------------------------------------------------------------------
#------------------------------------------FUNCTIONS-------------------------------------
#----------------------------------------------------------------------------------------
# Set random seed for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  
# Set GPU for TensorFlow if available
def set_gpu_if_exist(gpu):
    if gpu != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Use the first GPU and enable memory growth
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f'Using GPU {gpu} - {gpus[0].name}')
            except RuntimeError as e:
                print(e)
        else:
            print('No GPU found... Using CPU')
    else:
        print('Using CPU')

# Create a directory if it does not exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Initialize directories for saving models, plots, and results
def init_dirs(path=os.getcwd()):
    create_directory(os.path.join(path, 'models'))
    create_directory(os.path.join(path, 'plots'))
    create_directory(os.path.join(path, 'results'))

# Save results to a text file
def save_results_to_file(results, filename, is_dataframe=False):
    with open(filename, 'w') as f:
        if is_dataframe:
            f.write(results.to_string(index=True) + '\n')
        else:
            for key, value in results.items():
                f.write(f"{key.capitalize()}: {value}\n")

# Check if required packages are installed and install missing ones
def check_and_install_packages():
    required_packages = [
        'numpy', 'tensorflow', 'matplotlib', 'opencv-python', 
        'scikit-learn', 'ripser', 'persim', 'pandas', 
        'seaborn', 'tqdm'
    ]
    
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    
    for package in required_packages:
        if package not in installed_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Installed {package}")
            except Exception as e:
                print(f"Failed to install {package}: {e}")

    print("All required packages are installed.")

# Read, resize, and return images and their corresponding labels
def read_and_resize_images(image_files, labels, new_size):
    images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        images.append(img)
    
    return np.array(images), np.array(image_files), np.array(labels)

# Plot and save the confusion matrix as a heatmap
def plot_confusion_matrix(y_true, y_pred, accuracy, save_dir, title):
    save_dir = os.path.join(save_dir, 'plots')
    create_directory(save_dir)
    file_name = title.replace(' ', '_').lower()
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))  # Further increased figure size to accommodate larger colorbar

    # Create heatmap and store the returned colorbar object
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 40})
      
    # Increase colorbar label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=40)  # Increase colorbar tick label size
    plt.ylabel('Actual', fontsize=40, labelpad=20)
    plt.xlabel('Predicted', fontsize=40, labelpad=20)
    plt.tick_params(labelsize=40)  # Increased tick label size
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{file_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Compute metrics such as accuracy, precision, recall, and f1 score
def compute_metrics(y_true, y_pred, filename=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    metrics = {
        'accuracy': np.round(accuracy, 4),
        'precision': np.round(precision, 4),
        'recall': np.round(recall, 4),
        'f1': np.round(f1, 4)
    }

    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value}")
        
    # Save the metrics to a text file
    save_results_to_file(metrics, filename)
    
    return metrics
