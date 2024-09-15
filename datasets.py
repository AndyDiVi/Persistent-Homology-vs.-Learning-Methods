#----------------------------------------------------------------------------------------
#-------------------------------------------PACKAGES-------------------------------------
#----------------------------------------------------------------------------------------

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import tensorflow_datasets as tfds
from utilities import *
from persistent_homology_functions import *
from model_functions import * 



#----------------------------------------------------------------------------------------
#--------------------------------DATASET CLASS FOR CRACKS DATASET------------------------
#----------------------------------------------------------------------------------------

class CracksDataset:
    def __init__(self, data_dir, image_size=(64, 64), num_workers=os.cpu_count()):
        """
        Initialize the Cracks dataset with a specified directory, image size, and number of workers for parallel processing.
        """
        self.dataset_name = "Cracks"  # Name of the dataset
        self.data_dir = os.path.join(data_dir, self.dataset_name)  # Full path to the dataset directory
        self.image_size = image_size  # Target size to which images will be resized
        self.num_workers = num_workers // 2  # Use half the available CPU cores for processing
        self.positive_dir = os.path.join(self.data_dir, 'Positive')  # Directory for positive class images (cracks)
        self.negative_dir = os.path.join(self.data_dir, 'Negative')  # Directory for negative class images (no cracks)
        self.files, self.names, self.labels = self._load_data()  # Load all the image files and labels

    def _load_data(self):
        """
        Loads positive and negative images and labels them.
        """
        positive_files = [os.path.join(self.positive_dir, f) for f in os.listdir(self.positive_dir) if f.endswith(('.jpg', '.png'))]
        negative_files = [os.path.join(self.negative_dir, f) for f in os.listdir(self.negative_dir) if f.endswith(('.jpg', '.png'))]
        
        # Concatenate positive and negative files into a single list
        files = positive_files + negative_files
        # Labels: 1 for positive (cracks), 0 for negative (no cracks)
        labels = np.concatenate([np.ones(len(positive_files)), np.zeros(len(negative_files))])
        # Extract just the names of the files
        names = [os.path.basename(file) for file in files]
        
        # Shuffle the data (files, names, and labels) in unison
        files, names, labels = shuffle_data(files, names, labels)
        
        return np.array(files), np.array(names), np.array(labels)

    def get_train_val_test_split(self, train_size=0.6, val_size=0.2, random_seed=42):
        """
        Split the dataset into training, validation, and test sets based on the given sizes.
        """
        # Split data into training+validation and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.files, self.labels, test_size=(1 - train_size - val_size), stratify=self.labels, random_state=random_seed
        )
        # Further split training+validation into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=(val_size / (train_size + val_size)), stratify=y_train_val, random_state=random_seed
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_subset(self, files, labels, num_samples):
        """
        Get a balanced subset of positive and negative samples.
        """
        # Get indices for positive and negative samples
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        # Randomly select half positive and half negative samples
        pos_samples = np.random.choice(pos_indices, num_samples // 2, replace=False)
        neg_samples = np.random.choice(neg_indices, num_samples // 2, replace=False)

        # Combine and shuffle the indices
        subset_indices = np.concatenate((pos_samples, neg_samples))
        np.random.shuffle(subset_indices)

        # Return the subset of files and corresponding labels
        return np.array(files)[subset_indices], labels[subset_indices]

    def process_single_image(self, args):
        """
        Process a single image, resize, and binarize it.
        """
        file, label, dest_dir = args
        processed_filename = os.path.join(dest_dir, f"processed_{os.path.basename(file)}")

        if os.path.exists(processed_filename):
            return processed_filename, label

        # Read the image in grayscale mode
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # Binarize the image using a fixed threshold
        _, binary_img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
        # Add padding and resize the image
        binary_img = cv2.copyMakeBorder(binary_img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
        binary_img = cv2.resize(binary_img, self.image_size, interpolation=cv2.INTER_AREA)
        # Save the processed image
        cv2.imwrite(processed_filename, binary_img)

        return processed_filename, label

    def process_images(self, files, labels, dest_dir):
        """
        Process all images using parallel processing.
        """
        processed_dir = os.path.join(dest_dir, "Processed")
        os.makedirs(processed_dir, exist_ok=True)

        # Create argument list for parallel execution
        args_list = [(file, label, processed_dir) for file, label in zip(files, labels)]
        # Use ProcessPoolExecutor to parallelize image processing
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(executor.map(self.process_single_image, args_list), total=len(args_list), desc="Processing images"))

        # Extract processed files and labels
        processed_files, processed_labels = zip(*results)
        executor.shutdown()  # Close the executor
        return np.array(processed_files), np.array(processed_labels)

#----------------------------------------------------------------------------------------
#--------------------------------DATASET CLASS FOR MALARIA DATASET-----------------------
#----------------------------------------------------------------------------------------

class MalariaDataset:
    def __init__(self, data_dir=None, image_size=(64, 64), num_workers=os.cpu_count()):
        """
        Initialize the Malaria dataset with a specified directory, image size, and number of workers for parallel processing.
        """
        self.dataset_name = "malaria"
        if data_dir is None:
            data_dir = os.path.join(os.getcwd(), "datasets")
        self.data_dir = os.path.join(data_dir, self.dataset_name) 
        self.image_size = image_size
        self.num_workers = num_workers // 2
        self.files, self.names, self.labels = self._load_data()

    def _load_data(self):
        """
        Loads the Malaria dataset from disk or TensorFlow datasets if not found locally.
        """
        # If dataset already exists locally in 'raw' directory, load it
        if os.path.exists(f"{self.data_dir}/raw"):
            files, labels, names = [], [], []
            
            # Load the images from the local directory
            for file in os.listdir(f"{self.data_dir}/raw"):
                if file.endswith(('.jpg', '.png')):
                    files.append(os.path.join(f"{self.data_dir}/raw", file))
                    # Assign labels based on the filename (1 = parasitized, 0 = uninfected)
                    labels.append(0 if 'uninfected' in file else 1)
                    names.append(os.path.basename(file))

            # Shuffle the data (files, names, labels)
            files, names, labels = shuffle_data(files, names, labels)
            return np.array(files), np.array(names), np.array(labels)

        # Create 'raw' directory if it doesn't exist
        os.makedirs(f"{self.data_dir}/raw", exist_ok=True)

        # Load Malaria dataset from TensorFlow datasets
        ds = tfds.load(self.dataset_name, split='train', as_supervised=True)

        files, labels, names = [], [], []

        # Save each image and assign it a unique name
        for index, (img, label) in enumerate(tqdm(tfds.as_numpy(ds), total=len(ds), desc="Loading images")):
            name = f"image_{'uninfected_Negative' if label == 1 else 'parasitized_Positive'}_{index}.png"
            file = os.path.join(self.data_dir, "raw", name)
            cv2.imwrite(file, img)  # Save image to disk
            
            names.append(name)
            files.append(file)
            labels.append(label)
        
        # Inverse the label (since 1 should be uninfected and 0 parasitized)
        labels = np.array([0 if label == 1 else 1 for label in labels])

        # Shuffle the dataset
        files, names, labels = shuffle_data(files, names, labels)
        del ds  # Free up memory

        return files, names, labels

    def get_train_val_test_split(self, train_size=0.6, val_size=0.2, random_seed=42):
        """
        Split the dataset into training, validation, and test sets.
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.files, self.labels, test_size=(1 - train_size - val_size), stratify=self.labels, random_state=random_seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=(val_size / (train_size + val_size)), stratify=y_train_val, random_state=random_seed
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_subset(self, files, labels, num_samples):
        """
        Get a balanced subset of positive and negative samples.
        """
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        # Randomly select half positive and half negative samples
        pos_samples = np.random.choice(pos_indices, num_samples // 2, replace=False)
        neg_samples = np.random.choice(neg_indices, num_samples // 2, replace=False)

        # Combine and shuffle the indices
        subset_indices = np.concatenate((pos_samples, neg_samples))
        np.random.shuffle(subset_indices)

        return np.array(files)[subset_indices], labels[subset_indices]

    def process_single_image(self, args):
        """
        Process a single image: apply Gaussian blur, mask specific colors, and enhance features.
        """
        file, label, dest_dir = args
        processed_filename = os.path.join(dest_dir, f"processed_{os.path.basename(file)}")

        if os.path.exists(processed_filename):
            return processed_filename, label
        
        # Read the image
        img = cv2.imread(file)
        
        # Apply a Gaussian blur to smooth the image
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Convert the image to HSV color space for color-based filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define color range for malaria parasites (purple/pink)
        lower = np.array([100, 100, 100])  # Dark purple
        upper = np.array([170, 255, 255])  # Bright pink
        
        # Create a binary mask where purple/pink areas are white, and others are black
        mask = cv2.inRange(hsv, lower, upper)
        
        # Perform morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
        
        # Enhance small features using dilation
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black image to store the result
        result = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Fill the contours on the black image (this highlights the regions of interest)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:  # Only consider non-zero area contours
                cv2.drawContours(result, [contour], 0, (255), thickness=cv2.FILLED)

        # Add padding and resize the processed image
        result = cv2.copyMakeBorder(result, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
        result = cv2.resize(result, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Save the processed image
        cv2.imwrite(processed_filename, result)

        return processed_filename, label

    def process_images(self, files, labels, dest_dir):
        """
        Process all images using parallel processing.
        """
        processed_dir = os.path.join(dest_dir, "Processed")
        os.makedirs(processed_dir, exist_ok=True)

        # Create argument list for parallel execution
        args_list = [(file, label, processed_dir) for file, label in zip(files, labels)]
        # Use ProcessPoolExecutor to parallelize image processing
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(executor.map(self.process_single_image, args_list), total=len(args_list), desc="Processing images"))

        # Extract processed files and labels
        processed_files, processed_labels = zip(*results)
        executor.shutdown()  # Close the executor
        return np.array(processed_files), np.array(processed_labels)
