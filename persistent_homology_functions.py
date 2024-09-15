#----------------------------------------------------------------------------------------
#-------------------------------------------PACKAGES-------------------------------------
#----------------------------------------------------------------------------------------

# Import necessary packages for file handling, numerical operations, image processing, 
# persistence diagram computation, and multiprocessing
import os
import numpy as np
import cv2
import pickle
from ripser import ripser
from persim import plot_diagrams, PersistenceImager, bottleneck, bottleneck_matching
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from persim.images_weights import linear_ramp

# Import custom functions
from utilities import *

#----------------------------------------------------------------------------------------
#---------------------------------PERSISTENCE DIAGRAM UTILITIES--------------------------
#----------------------------------------------------------------------------------------

# Function to save a persistence diagram to disk
def save_persistence_diagram(diagram, directory, base_filename):
    pickle_path = os.path.join(directory, f"{base_filename}_diagram.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(diagram, f)

# Function to load a persistence diagram from disk
def load_persistence_diagram(filepath):
    with open(filepath, 'rb') as f:
        diagram = pickle.load(f)
    return diagram

# Compute persistence diagrams for an image using an optimized process
def compute_persistence_diagrams_optimized(img):
    try:
        points = np.column_stack(np.where(img == 0))
        diagrams = ripser(points, maxdim=1)['dgms']
        return diagrams
    except Exception as e:
        print(f"Error computing persistence diagrams: {e}")
        return None

#----------------------------------------------------------------------------------------
#------------------------------COMPUTE AND LOAD DIAGRAMS---------------------------------
#----------------------------------------------------------------------------------------

# Function to compute or load persistence diagrams for each image
def compute_load_diagrams(args):
    file, label, diagram_directory = args
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    base_filename = os.path.splitext(os.path.basename(file))[0]
    pickle_path = os.path.join(diagram_directory, f"{base_filename}_diagram.pkl")
    if os.path.exists(pickle_path):
        diagram = load_persistence_diagram(pickle_path)
    else:
        diagram = compute_persistence_diagrams_optimized(img)
        if diagram is not None:
            save_persistence_diagram(diagram, diagram_directory, base_filename)
            # Apply custom weight to highlight features with persistence > 2.5
            weight_params = {'low': 0.0, 'high': 1.0, 'start': 2.5, 'end': 20.0}
            
            plot_and_save_persistent_features(diagram, diagram_directory, base_filename, weight_function=linear_ramp, weight_params=weight_params)
            plt.close('all')
        else:
            print(f"Could not compute diagram for {file}")
    return file, diagram, label

# Optimized function to compute homology diagrams for a batch of images
def compute_homology_optimized(processed_files, labels, base_dir, num_workers=None):
    assert len(processed_files) == len(labels), "Number of files and labels must match"
    diagram_directory = os.path.join(base_dir, "Diagrams")
    os.makedirs(diagram_directory, exist_ok=True)
    args_list = [(file, label, diagram_directory) for file, label in zip(processed_files, labels)]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(compute_load_diagrams, args_list), total=len(args_list), desc="Processing images"))
    executor.shutdown()
    diagrams_names, diagrams, diagrams_labels = zip(*results)

    # Handle cases where diagram computation failed (None values)
    valid_diagrams = []
    valid_names = []
    valid_labels = []
    for name, diagram, label in zip(diagrams_names, diagrams, diagrams_labels):
        if diagram is not None:
            valid_diagrams.append(diagram)
            valid_names.append(name)
            valid_labels.append(label)
        else:
            print(f"Diagram computation failed for image: {name}")

    return valid_diagrams, valid_names, valid_labels


#----------------------------------------------------------------------------------------
#------------------------------PERSISTENCE DIAGRAM PLOTTING------------------------------
#----------------------------------------------------------------------------------------

# Function to plot and save persistence diagrams for H0 and H1
def plot_persistence_diagrams(diagrams, base_filename):
    
    ## Plot and save H0 diagram
    #plt.figure(figsize=(8, 6))
    #plot_diagrams(diagrams[0], labels="H_0", show=False)
    #plt.title('H0 Persistent Diagram', fontsize=16)
    #plt.tight_layout()
    #plt.savefig(f"{base_filename}_H0_diagram.png")
    #plt.close()
    
    # Plot and save H1 diagram
    plt.figure(figsize=(8, 6))
    plot_diagrams(diagrams[1], labels="H_1", show=False)
    plt.title('H1 Persistent Diagram', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{base_filename}_H1_diagram.png")
    plt.close()


# Function to plot persistence images with a weight function applied
def plot_persistence_images(diagram, image_filename, weight_function=linear_ramp, weight_params=None):
    pimgr = PersistenceImager(pixel_size=0.5, birth_range=(0, 50), pers_range=(0, 50))
    
    if weight_function is not None:
        pimgr.weight = weight_function
        if weight_params is not None:
            pimgr.weight_params = weight_params
    
    # Filter out invalid points
    valid_diagram = []
    for point in diagram:
        if np.isfinite(point[0]) and np.isfinite(point[1]):
            valid_diagram.append(point)
    
    if len(valid_diagram) == 0:
        print(f"Warning: No valid points in diagram for {image_filename}")
        return
    
    valid_diagram = np.array(valid_diagram)
    
    # Plot the persistence image
    plt.figure(figsize=(8, 6))
    pimgr.plot_image(pimgr.transform(valid_diagram))
    plt.title('Persistence Image Highlighting Persistence > 2.5', fontsize=16)
    plt.tight_layout()
    plt.savefig(image_filename)
    plt.clf()


# Combined function to generate and save persistence diagrams and persistence images
def plot_and_save_persistent_features(diagram, directory, base_filename, weight_function=linear_ramp, weight_params=None):
    os.makedirs(os.path.join(directory, "diagrams"), exist_ok=True)
    os.makedirs(os.path.join(directory, "pers_imgs"), exist_ok=True)

    # Generate and save H0 and H1 diagrams
    plot_persistence_diagrams(diagram, os.path.join(directory, "diagrams", f"{base_filename}"))

    # Generate and save H0 persistence image
    #plot_persistence_images(diagram[0], os.path.join(directory, "pers_imgs", f"{base_filename}_H0_persim.png"), weight_function, weight_params)

    # Generate and save H1 persistence image
    plot_persistence_images(diagram[1], os.path.join(directory, "pers_imgs", f"{base_filename}_H1_persim.png"), weight_function, weight_params)

def plot_H1_diagram(diagram, diagram_filename):
    plt.figure(figsize=(8, 6))
    plot_diagrams(diagram[1], labels="H_1", show=False)
    plt.title('Persistent Diagram', fontsize=16)
    plt.tight_layout()
    plt.savefig(diagram_filename)
    plt.close()
    
#----------------------------------------------------------------------------------------
#--------------------------------BOTTLENECK DISTANCE-------------------------------------
#----------------------------------------------------------------------------------------
def classify_images_by_num_holes(base_dir, diagrams, true_labels, image_names, threshold, num_holes_threshold):
    predicted_labels = []
    for diagram, true_label, image_name in zip(diagrams, true_labels, image_names):
        num_holes = sum(1 for interval in diagram[1] if interval[1] - interval[0] > threshold)
        predicted_label = 1 if num_holes >= num_holes_threshold else 0
        predicted_labels.append(predicted_label)
        log_misclassification(true_label, predicted_label, image_name, base_dir, 'num_holes')
    return true_labels, predicted_labels

# Compute bottleneck distances from a reference diagram
def compute_bottleneck_distances_from_reference(args):
    reference_diagram, diagram, filename, true_label, image_name = args
    distance, matching = bottleneck(reference_diagram[1], diagram[1], matching=True)
    return distance, true_label, image_name

# Compute bottleneck distances and classify images based on the threshold
def classify_images_by_bottleneck_distance(base_dir, diagrams, true_labels, image_names, reference_diagram):
    # Prepare arguments for parallel processing
    args_list = [(reference_diagram, diagram, 
                  os.path.join(base_dir, 'plots', f"{os.path.splitext(image_name)[0]}_bottleneck.png"),
                  true_label, image_name)
                 for diagram, true_label, image_name in zip(diagrams, true_labels, image_names)]

    # Determine the number of processes to use
    max_workers = os.cpu_count() // 2

    distances = []
    processed_true_labels = []
    
    # Prepare file for writing distances
    distances_file = os.path.join(base_dir, 'bottleneck_distances.csv')
    with open(distances_file, 'w') as f:
        f.write("Image Name,True Label,Predicted Label,Bottleneck Distance\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(compute_bottleneck_distances_from_reference, args): args 
                          for args in args_list}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_args), total=len(future_to_args), desc="Computing bottleneck distances"):
            try:
                distance, true_label, image_name = future.result()
                distances.append(distance)
                processed_true_labels.append(true_label)
                
                # We'll write the predicted label after we've calculated the threshold
            except Exception as exc:
                print(f'An image generated an exception: {exc}')

        # Determine threshold using median distance
        threshold = np.median(distances)
        predicted_labels = [0 if d <= threshold else 1 for d in distances]

        # Now we can write the results to the file and log misclassifications
        for distance, true_label, predicted_label, image_name in zip(distances, processed_true_labels, predicted_labels, image_names):
            # Write distance to file
            with open(distances_file, 'a') as f:
                f.write(f"{image_name},{true_label},{predicted_label},{distance} - Threshold: {threshold}\n")
                
            # Log misclassification
            if true_label != predicted_label:
                log_misclassification(true_label, predicted_label, image_name, base_dir, 'bottleneck')
    
    executor.shutdown()
    return distances, processed_true_labels, predicted_labels


# LOG MISCLASSIFICATION (FALSE POSITIVE OR FALSE NEGATIVE)
def log_misclassification(true_label, predicted_label, image_name, base_dir, method):
    if true_label != predicted_label:
        misclassification_type = 'false_negative' if true_label == 1 else 'false_positive'
        filename = f'{base_dir}/results/{method}_{misclassification_type}.txt'
        with open(filename, 'a') as f:
            log_message = f"{image_name}"
            f.write(log_message + "\n")
        #print(f"Diagram {image_name} is a {misclassification_type}")

def read_features_img(args, imgs_dir, img_size):
    file, label = args
    #h0_path = os.path.join(imgs_dir, "pers_imgs", f"processed_{os.path.splitext(os.path.basename(file))[0]}_H0_persim.png")
    h1_path = os.path.join(imgs_dir, "pers_imgs", f"processed_{os.path.splitext(os.path.basename(file))[0]}_H1_persim.png")
    
    #h0_img = cv2.imread(h0_path, cv2.IMREAD_GRAYSCALE)
    h1_img = cv2.imread(h1_path, cv2.IMREAD_GRAYSCALE)
    
    if h1_img is None:
        raise ValueError(f"Image at {h1_path} could not be read.")
    
    #h0_img = cv2.resize(h0_img, img_size)
    h1_img = cv2.resize(h1_img, img_size)
    
    #combined_img = np.concatenate((h0_img, h1_img), axis=1)
    
    return h1_img, file, label

def read_features_imgs(files, labels, imgs_dir, img_size=(64, 64), num_workers=os.cpu_count()):
    imgs_dir = os.path.join(imgs_dir, "Diagrams")

    args_list = [(file, label) for file, label in zip(files, labels)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(partial(read_features_img, imgs_dir=imgs_dir, img_size=img_size), args_list), 
                            total=len(args_list), desc="Reading persistent images"))

    features, features_names, labels = zip(*results)
    
    return np.array(features), np.array(features_names), np.array(labels)