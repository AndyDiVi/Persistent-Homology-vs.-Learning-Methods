#----------------------------------------------------------------------------------------
#-------------------------------------------PACKAGES-------------------------------------
#----------------------------------------------------------------------------------------

# Import custom functions for handling utilities, models, and persistent homology
from utilities import *
from model_functions import *
from persistent_homology_functions import *
from datasets import *
import os
import time

import matplotlib
matplotlib.rcParams.update({
    "text.usetex": False,
    "text.latex.preamble": ""
})
matplotlib.use('Agg')  # Use Agg backend to avoid LaTeX dependency

# Import tools for persistent homology and bottleneck distance from Ripser and Persim
from ripser import ripser
from persim import plot_diagrams, PersistenceImager, bottleneck, bottleneck_matching
from persim.landscapes import (
    PersLandscapeExact, PersLandscapeApprox, plot_landscape_simple, plot_landscape, snap_pl
)

# Handle interruption (CTRL+C) gracefully during execution
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Main function to execute experiments based on the dataset, chosen approach, and number of training samples
def main(dataset, approach, training_samples):
    # Prepare dataset and directories for saving results
    dataset_name = dataset.dataset_name
    base_dir = os.path.join(os.getcwd(), f"experiments_{dataset_name}", f"approach_{approach}", f"training_samples_{training_samples}")
    data_dir = dataset.data_dir
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")
    init_dirs(base_dir)  # Initialize necessary directories
    num_workers = os.cpu_count() // 2  # Utilize half of the available CPU cores for multiprocessing

    # Step 1: Split the dataset into training, validation, and test sets
    print(f"-------------- Creating training, validation and test sets --------------")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_train_val_test_split()
    
    # Ensure the image-label pairs are correct
    check_images_labels_pairs(X_train, y_train)
    check_images_labels_pairs(X_val, y_val)
    check_images_labels_pairs(X_test, y_test)

    # Subset the training data if specified by the user (useful for limited labeled data experiments)
    X_train, y_train = dataset.get_subset(X_train, y_train, training_samples)
    X_test, y_test = dataset.get_subset(X_test, y_test, training_samples)

    check_images_labels_pairs(X_train, y_train)
    check_images_labels_pairs(X_test, y_test)

    # Step 2: Process images for analysis
    print(f"-------------- Processing images --------------")
    X_train_processed, y_train_processed = dataset.process_images(X_train, y_train, train_dir)
    X_test_processed, y_test_processed = dataset.process_images(X_test, y_test, test_dir)

    check_images_labels_pairs(X_train_processed, y_train_processed)
    check_images_labels_pairs(X_test_processed, y_test_processed)

    # Step 3: Compute persistence homology diagrams for the processed images
    print(f"-------------- Computing homology --------------")
    if approach in [1, 2, 4]:
        X_train_diagrams, train_diagrams_names, train_diagrams_labels = compute_homology_optimized(X_train_processed, y_train_processed, train_dir, num_workers=num_workers)
        X_test_diagrams, test_diagrams_names, test_diagrams_labels = compute_homology_optimized(X_test_processed, y_test_processed, test_dir, num_workers=num_workers)

        # Verify the diagram names and labels match
        check_images_labels_pairs(train_diagrams_names, train_diagrams_labels)
        check_images_labels_pairs(test_diagrams_names, test_diagrams_labels)

        # Shuffle the data for better randomization during training
        print(f"-------------- Shuffling data --------------")
        shuffle_data(X_train_diagrams, train_diagrams_names, train_diagrams_labels)
        shuffle_data(X_test_diagrams, test_diagrams_names, test_diagrams_labels)

    # Step 4: Perform classification based on the selected approach
    print(f"-------------- Classification --------------")

    # Approach 1: Classify by the number of persistent homology "holes" (H1 features)
    if approach == 1:
        print("Starting Classification by number of holes...")
        # Define dataset-specific thresholds for classification
        threshold = 3 if dataset_name.lower() == "cracks" else 1.5
        num_holes_threshold = 2 if dataset_name.lower() == "cracks" else 1

        # Perform classification and compute metrics
        true_labels, predicted_labels = classify_images_by_num_holes(
            base_dir, X_test_diagrams, test_diagrams_labels, test_diagrams_names, threshold=threshold, num_holes_threshold=num_holes_threshold
        )
        metrics = compute_metrics(true_labels, predicted_labels, f"{base_dir}/results/metrics.txt")
        plot_confusion_matrix(true_labels, predicted_labels, metrics['accuracy'], base_dir, f"Approach {approach}")

    # Approach 2: Classify by the bottleneck distance from a reference persistence diagram
    elif approach == 2:
        print("Starting Classification by bottleneck distance from a reference...")
        negative_diagrams = [X_test_diagrams[i] for i in range(len(X_test_diagrams)) if test_diagrams_labels[i] == 0]
        reference_diagram = negative_diagrams[0]
        
        # Plot the reference persistence diagram
        #plot_H1_diagram(reference_diagram, f"{base_dir}/results/reference_diagram.png")

        # Classify based on bottleneck distance and compute metrics
        distances, true_labels, predicted_labels = classify_images_by_bottleneck_distance(
            base_dir, X_test_diagrams, test_diagrams_labels, test_diagrams_names, reference_diagram
        )
        with open(f"{base_dir}/results/distances_{training_samples}.txt", "w") as f:
            for i, (distance, true_label, predicted_label) in enumerate(zip(distances, true_labels, predicted_labels)):
                f.write(f"Image {i}: distance={distance}, true_label={true_label}, predicted_label={predicted_label}\n")
        metrics = compute_metrics(true_labels, predicted_labels, f"{base_dir}/results/metrics.txt")
        plot_confusion_matrix(true_labels, predicted_labels, metrics['accuracy'], base_dir, f"Approach {approach}")

    # Approach 3: Classify using traditional machine learning (e.g., Logistic Regression, Random Forest, SVM)
    elif approach == 3:
        print("Starting Classification using a simple ML model (Logistic Regression)...")
        # Resize and prepare the image data
        X_train_images, train_images_names, _ = read_and_resize_images(X_train, y_train, dataset.image_size)
        X_test_images, test_images_names, _ = read_and_resize_images(X_test, y_test, dataset.image_size)
        
        methods = ['logistic_regression', 'random_forest', 'svm']
        for method in methods:
            y_pred, probs = classifier_ml(base_dir, X_train_images, y_train, X_test_images, method=method)
            metrics = compute_metrics(y_test, y_pred, f"{base_dir}/results/{method}_metrics.txt")
            plot_confusion_matrix(y_test, y_pred, metrics['accuracy'], base_dir, f"Approach {approach} - {method}")

    # Approach 4: Classify using features from persistent images and machine learning models
    elif approach == 4:
        print("Starting Classification by Persistent Images Features in ML...")
        # Extract and reshape features for classification
        X_train_features, train_features_names, train_features_labels = read_features_imgs(X_train, y_train, train_dir, img_size=dataset.image_size, num_workers=num_workers)
        X_test_features, test_features_names, test_features_labels = read_features_imgs(X_test, y_test, test_dir, img_size=dataset.image_size, num_workers=num_workers)

        X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
        X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

        methods = ['logistic_regression', 'random_forest', 'svm']
        for method in methods:
            y_pred, probs = classifier_ml(base_dir, X_train_features, train_features_labels, X_test_features, method=method)
            metrics = compute_metrics(test_features_labels, y_pred, f"{base_dir}/results/{method}_metrics.txt")
            plot_confusion_matrix(test_features_labels, y_pred, metrics['accuracy'], base_dir, f"Approach {approach} - {method}")
    else:
        print("Invalid approach number")


# %%

if __name__ == "__main__":


    # Set random seed for reproducibility
    RANDOM_SEED = 42
    set_random_seed(RANDOM_SEED)

    # Set GPU if available
    set_gpu_if_exist(-1)

    # Define the available approaches
    approaches_dict = {
        1: "Classification by number of holes",
        2: "Classification by bottleneck distance from a reference",
        3: "Classification using a simple ML model (Logistic Regression)",
        4: "Classification by Persistent Images Features in ML"        
    }

    # Print the available approaches
    print("Choose the approach to use:")
    for key, value in approaches_dict.items():
        print(f"{key}: {value}")

    # Ask for the approach number
    approach = int(input(f"Enter the number of the approach to use: \t"))

    # Check if the approach is valid
    assert approach in approaches_dict.keys(), "Invalid approach number"

    if approach == 1 or approach == 2:
        samples_list = [2]
    else:
        samples_list = [2, 10, 20, 50, 100, 200, 400, 600, 800, 1000]
    
    dataset_name = input("Enter the dataset name to use ('cracks', 'malaria'): \t")
    

    if dataset_name.lower() == "cracks":
        cracks_dataset = CracksDataset(data_dir='/datasets')
        dataset = cracks_dataset
    elif dataset_name.lower() == "malaria":
        malaria_dataset = MalariaDataset()
        dataset = malaria_dataset
    else:
        raise ValueError("Invalid dataset name")
    
    # Run the selected approach
    for i in samples_list:
        print(f"\n\n**************************************** Training samples: {i}")
        start_time = time.time()
        main(dataset, approach, i)
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
