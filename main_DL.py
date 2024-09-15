#----------------------------------------------------------------------------------------
#-------------------------------------------PACKAGES-------------------------------------
#----------------------------------------------------------------------------------------

from utilities import *
from model_functions import *
from datasets import *
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "text.usetex": False,
    "text.latex.preamble": ""
})
matplotlib.use('Agg')  
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, AdamW
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


#----------------------------------------------------------------------------------------
#-----------------------------------DEEP LEARNING MAIN-----------------------------------
#----------------------------------------------------------------------------------------


# Set random seed for reproducibility
RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)

# Ask for which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = input("Enter the GPU number to use: ")

# Set GPU if available
set_gpu_if_exist(0)

# Ask for the dataset to use
dataset_name = input("Enter the dataset to use (Cracks/Malaria): ")
if dataset_name.lower() == 'cracks':
    dataset = CracksDataset(data_dir='/Datasets/Classification')
elif dataset_name.lower() == 'malaria':
    dataset = MalariaDataset()
else:
    raise ValueError("Invalid dataset name. Please enter 'Cracks' or 'Malaria'.")

data_dir = dataset.data_dir
dataset_name = dataset.dataset_name

# Create the training and test sets
print(f"-------------- Creating training, validation and test sets --------------")
X_train_original, X_val_original, X_test_original, y_train_original, y_val_original, y_test_original = dataset.get_train_val_test_split()

check_images_labels_pairs(X_train_original, y_train_original)
check_images_labels_pairs(X_val_original, y_val_original)
check_images_labels_pairs(X_test_original, y_test_original)

X_test, y_test = X_test_original, y_test_original
X_val, y_val = X_val_original, y_val_original
print(f"Training samples: {len(X_train_original)} - Validation samples: {len(X_val)} - Test samples: {len(X_test)}")
print(f"-------------------------------------------------------------------------")

models_list = ['VGG16', 'ResNet50', 'InceptionV3', 'MobileNetV2']

imagenet_weights = True
training_samples_list = [2, 4, 6, 8, 10, 20, 50, 100, 200, 400, 600, 800, 1000] # Comprise both classes
epochs = 100

# Initialize a dictionary to store the metrics for each model (imagenet or random weights) and number of training samples
metrics = {f"{model_architecture}_{'imagenet' if imagenet_weights else 'random'}": {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for model_architecture in models_list}

for model_architecture in models_list:
    
    if model_architecture == 'InceptionV3':
        input_shape = (75, 75, 3)
    else:
        input_shape = (64, 64, 3)
            
    for num_samples in training_samples_list:

        # Build and compile the model
        model = build_deep_learning_model(input_shape, model_name=model_architecture, imagenet_weights=imagenet_weights, dataset_name=dataset_name)
        model_name = f"{model_architecture}_{'imagenet' if imagenet_weights else 'random'}"
        
        # Create the base directory for this model
        base_dir = os.path.join(os.getcwd(), f"experiments_{dataset_name}", "deep_learning", f"{model_name}")
        # Initialize the directories to store the models and the results
        init_dirs(base_dir)
        
        # Set the batch size based on the number of samples in the training set 
        batch_size = 2 if num_samples <= 10 else 4 if num_samples <= 50 else 8 if num_samples <= 100 else 16
        
        # Get a subset of the training set
        X_train, y_train = dataset.get_subset(X_train_original, y_train_original, num_samples)
        check_images_labels_pairs(X_train, y_train)

        # Create generators for this subset
        train_generator, validation_generator, test_generator = create_generators(
            X_train, X_val, X_test, 
            y_train, y_val, y_test,
            input_shape=input_shape, batch_size=batch_size
        )
        print(f"\n\n ------------------ \n\n")
        print(f"\n\nTraining and evaluating the model {model_name} for {num_samples} total samples...")
        print(f"Parameters: {model.count_params()}")
        print(f"Training samples: {train_generator.samples} - Validation samples: {validation_generator.samples} - Test samples: {test_generator.samples}")
        print(f"Training samples per class: {num_samples//2}")
        print(f"Input shape: {input_shape} - Batch size: {batch_size} - Epochs: {epochs}")
        print(f"\n\n ------------------ \n\n") 

        start_time = time.time()
        # Train and evaluate the model
        accuracy, precision, recall, f1 = train_and_evaluate_model(model, model_architecture, train_generator, validation_generator, test_generator, epochs=epochs, base_dir=base_dir)
        end_time = time.time()
        
        # Store the metrics
        metrics[model_name]['accuracy'].append(accuracy)
        metrics[model_name]['precision'].append(precision)
        metrics[model_name]['recall'].append(recall)
        metrics[model_name]['f1'].append(f1)
        
        # Log the metrics
        print(f"Model: {model_name}, Training samples per class: {num_samples//2} - Total time: {end_time - start_time} seconds")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Create a DataFrame for the metrics of this model
    model_metrics_df = pd.DataFrame(metrics[model_name], index=[f"{num_samples//2} samples" for num_samples in training_samples_list])

    # Save the metrics to a text file
    results_file = os.path.join(base_dir, 'results', f"{model_name}_training_results.txt")
    save_results_to_file(model_metrics_df, results_file, is_dataframe=True)

    print(f"\nMetrics for {model_name}:")
    print(model_metrics_df)

    # Plot the metrics for this model
    plt.figure(figsize=(12, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot([num_samples//2 for num_samples in training_samples_list], metrics[model_name][metric], label=metric.capitalize())
    plt.title(f'Metrics - {model_name}')
    plt.xlabel('Training samples')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'plots', f'{model_name}_metrics.png'))
    plt.show()


# Create a list of sample sizes (half of the total samples used in training)
samples_list = [num_samples//2 for num_samples in training_samples_list]

# Create a combined DataFrame for all models
combined_metrics_df = pd.concat([
    pd.DataFrame({
        'Model': model,
        'Samples': samples_list,
        'Accuracy': metrics[f"{model}_{'imagenet' if imagenet_weights else 'random'}"]["accuracy"],
        'Precision': metrics[f"{model}_{'imagenet' if imagenet_weights else 'random'}"]["precision"],
        'Recall': metrics[f"{model}_{'imagenet' if imagenet_weights else 'random'}"]["recall"],
        'F1': metrics[f"{model}_{'imagenet' if imagenet_weights else 'random'}"]["f1"]
    }) for model in models_list
])

# Reset the index to make sure it's continuous
combined_metrics_df = combined_metrics_df.reset_index(drop=True)

# Print the combined metrics
print("\nCombined Metrics for All Models:")
print(combined_metrics_df)

# Save the combined metrics to a text file
results_dir = os.path.join(os.getcwd(), f"{dataset_name}", "deep_learning", "results")
os.makedirs(results_dir, exist_ok=True)
combined_results_file = os.path.join(results_dir, f"combined_metrics_{'imagenet' if imagenet_weights else 'random'}.txt")
save_results_to_file(combined_metrics_df, combined_results_file, is_dataframe=True)


# Plot combined metrics
plt.figure(figsize=(15, 10))
for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
    plt.subplot(2, 2, i+1)
    for model in models_list:
        model = f"{model}_{'imagenet' if imagenet_weights else 'random'}"
        plt.plot(training_samples_list, metrics[model][metric], label=model)
    plt.title(f'{metric.capitalize()}')
    plt.xlabel('Training samples')
    plt.ylabel('Score')
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"combined_metrics_{'imagenet' if imagenet_weights else 'random'}.png"))
plt.show()
