#----------------------------------------------------------------------------------------
#-------------------------------------------PACKAGES-------------------------------------
#----------------------------------------------------------------------------------------

import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from utilities import *

#----------------------------------------------------------------------------------------
#------------------------------------------FUNCTIONS-------------------------------------
#----------------------------------------------------------------------------------------

def check_images_labels_pairs(files, labels):
    # Check if the number of files and labels are the same
    assert len(files) == len(labels), f"Number of files ({len(files)}) and labels ({len(labels)}) must be the same"
    
    # Check if the files exist
    for file in files:
        assert os.path.exists(file), f"File {file} does not exist"
        
    # Check if the labels are either 0 or 1
    assert set(labels) == {0, 1}, f"Labels must be either 0 or 1: {set(labels)}"
    
    # Check that if file contains 'Positive' then label is 1, otherwise label is 0
    for file, label in zip(files, labels):
        if 'Negative' in file or 'Non-Covid' in file:
            assert label == 0, f"File {file} contains 'Negative'/'Non-Covid' but has label {label}"
        else:
            assert label == 1, f"File {file} does not contain 'Negative'/'Non-Covid' but has label {label}"
    

def shuffle_data(X_files, X_names, y):
    combined = list(zip(X_files, X_names, y))
    random.shuffle(combined)
    X_files[:], X_names[:], y[:] = zip(*combined)
    return X_files, X_names, y
    

def create_generators(train_files, val_files, test_files, train_labels, val_labels, test_labels, input_shape, batch_size=32, val_split=0.2, random_seed=42):
    # Convert labels to strings for ImageDataGenerator
    train_labels = train_labels.astype(str)
    val_labels = val_labels.astype(str)
    test_labels = test_labels.astype(str)

    # Create a DataFrame for train, validation, and test sets
    train_df = pd.DataFrame({'filename': train_files, 'label': train_labels})
    val_df = pd.DataFrame({'filename': val_files, 'label': val_labels})
    test_df = pd.DataFrame({'filename': test_files, 'label': test_labels})

    # Image Data Generators with rescale to [0, 1]
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        #rotation_range=15,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #shear_range=0.1,
        #zoom_range=0.1,
        #horizontal_flip=True,
        #fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Train generator
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='label',
        target_size=(input_shape[0], input_shape[1]), # (height, width) 
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=random_seed
    )

    # Validation generator
    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='label',
        target_size=(input_shape[0], input_shape[1]), # (height, width)
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Test generator
    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        x_col='filename',
        y_col='label',
        target_size=(input_shape[0], input_shape[1]), # (height, width)
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator



#-----------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------DEEP LEARNING MODELS--------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------

#DEFINING MODEL-----------------------------------------------------------------
def build_deep_learning_model(input_shape, model_name='VGG16', imagenet_weights=True, dataset_name='cracks'):
    # Load the pre-trained model
    weights = 'imagenet' if imagenet_weights else None
    learning_rate = 0.0001 if imagenet_weights else 0.001
    
    if model_name == 'VGG16':
        base_model = VGG16(input_shape=input_shape, weights=weights, include_top=False)
        last_layer_name = 'block5_pool'
    elif model_name == 'ResNet50':
        base_model = ResNet50(input_shape=input_shape, weights=weights, include_top=False)
        last_layer_name = 'conv5_block3_out'
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(input_shape=input_shape, weights=weights, include_top=False)
        last_layer_name = 'mixed7'
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(input_shape=input_shape, weights=weights, include_top=False)
        last_layer_name = 'block_12_add'
    else:
        raise ValueError(f"Model {model_name} not found. Choose from 'VGG16', 'ResNet50', 'InceptionV3', 'MobileNetV2'")
    
    # Freeze the layers of the pre-trained model
    if imagenet_weights and dataset_name == 'cracks':
        for layer in base_model.layers:
            layer.trainable = False

    # Get the output of the last layer
    last_output = base_model.get_layer(last_layer_name).output

    # Add new layers
    x = Flatten()(last_output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    # Create the new model
    model = Model(base_model.input, x, name=model_name)
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # Print the summary of the model
    #model.summary()
    return model


def plot_model_history(history, file_name, title):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy - {title}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss - {title}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save the figure
    plt.savefig(f"{file_name}")
    plt.show()
    

#TRAINING MODEL ----------------------------------------------------------------   
def train_and_evaluate_model(model, model_name, train_generator, val_generator, test_generator, epochs=10, base_dir=os.getcwd()):
    
    # Get the number of samples in the training set
    num_samples = train_generator.samples 
    
    # Define the checkpoint filepath
    os.makedirs(f"{base_dir}/models", exist_ok=True)
    checkpoint_filepath = os.path.join(f"{base_dir}/models", f'best_{model_name}_{num_samples}samples.h5')
    
    if not os.path.exists(checkpoint_filepath):
        # Define the callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1),
            #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]
            
        # Train the model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Plot the history
        history_file_name = f"{base_dir}/plots/{model_name}_{num_samples}samples_history.png"
        plot_model_history(history, history_file_name, title=f"{model_name}_{num_samples}samples")

    # Load the best model
    model.load_weights(checkpoint_filepath)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy} - Test loss: {test_loss}")

    # Predict the test set
    y_pred = model.predict(test_generator).flatten()
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # Get the true labels
    y_true = test_generator.classes

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Round the metrics to 4 decimal places
    accuracy = round(accuracy, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    
    # Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred, accuracy, base_dir, title=f"{model_name}_{num_samples}samples")

    return accuracy, precision, recall, f1


#CROSS VALIDATION----------------------------------------------------------------
def perform_cross_validation(base_dir, model_name, train_files, train_labels, input_shape, batch_size=32, epochs=10, seed=42):

    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold = 1
    
    # Initialize lists for storing metrics for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Loop over the folds and train the model
    for train_index, val_index in kf.split(train_files):
        print(f"Training on fold {fold}...")
        
        train_fold_files, val_fold_files = train_files[train_index], train_files[val_index]
        train_fold_labels, val_fold_labels = train_labels[train_index], train_labels[val_index]

        # Build and compile the model
        model = build_deep_learning_model(input_shape, model_name=model_name, imagenet_weights=False)

        # Create the generators
        train_generator, val_generator, test_generator = create_generators(
            base_dir, train_fold_files, val_fold_files, train_fold_labels, val_fold_labels,
            input_shape=input_shape, batch_size=batch_size
        )

        # Train and evaluate the model
        accuracy, precision, recall, f1 = train_and_evaluate_model(model, model_name, train_generator, val_generator, test_generator, epochs=epochs, base_dir=base_dir)

        # Store the metrics
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
        # Log the metrics
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        fold += 1
        
    # Compute the mean and standard deviation of the metrics
    accuracy_mean = np.mean(accuracy_scores)
    accuracy_std = np.std(accuracy_scores)
    precision_mean = np.mean(precision_scores)
    precision_std = np.std(precision_scores)
    recall_mean = np.mean(recall_scores)
    recall_std = np.std(recall_scores)
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    
    # Log the mean and standard deviation of the metrics
    print(f"\n --- Cross Validation Results ---")
    print(f"Mean Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"Mean Precision: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"Mean Recall: {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"Mean F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
    
    return accuracy_scores, precision_scores, recall_scores, f1_scores    



#-------------------------------------------------------------------------------------------------------
#-------------------------------------MACHINE LEARNING MODELS----------------------------------------------
#---------------------------------------------------------------------------------------------------------
def flatten_images(X):
    return np.array([x.flatten() for x in X])

def flatten_and_scale(X):
    X = flatten_images(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# LOGISTIC REGRESSION
def logistic_regression(base_dir, X_train, y_train, X_test):
    X_train = flatten_and_scale(X_train)
    X_test = flatten_and_scale(X_test)

    model = LogisticRegression(max_iter=2000)
    
    if not os.path.exists(f"{base_dir}/models/logistic_regression_model.h5"):
        model.fit(X_train, y_train)
        joblib.dump(model, f"{base_dir}/models/logistic_regression_model.h5")
    else:
        model = joblib.load(f"{base_dir}/models/logistic_regression_model.h5")       
        
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    return y_pred, probs

# SUPPORT VECTOR MACHINE
def support_vector_machine(base_dir, X_train, y_train, X_test):
    X_train = flatten_and_scale(X_train)
    X_test = flatten_and_scale(X_test)
    
    model = SVC(probability=True)
    
    if not os.path.exists(f"{base_dir}/models/svm_model.h5"):
        model.fit(X_train, y_train)
        joblib.dump(model, f"{base_dir}/models/svm_model.h5")
    else:
        model = joblib.load(f"{base_dir}/models/svm_model.h5")
        
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)

    return y_pred, probs


# RANDOM FOREST
def random_forest(base_dir, X_train, y_train, X_test):
    X_train = flatten_and_scale(X_train)
    X_test = flatten_and_scale(X_test)

    model = RandomForestClassifier()

    if not os.path.exists(f"{base_dir}/models/random_forest_model.h5"):
        model.fit(X_train, y_train)
        joblib.dump(model, f"{base_dir}/models/random_forest_model.h5")
    else:
        model = joblib.load(f"{base_dir}/models/random_forest_model.h5")

    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    return y_pred, probs

# MACHINE LEARNING CLASSIFIER
def classifier_ml(base_dir, X_train, y_train, X_test, method):
    methods = {
        'logistic_regression': logistic_regression,
        'svm': support_vector_machine,
        'random_forest': random_forest
    }

    if method in methods:
        print(f"\n\nTraining and evaluating {method.replace('_', ' ').capitalize()}...")
        return methods[method](base_dir, X_train, y_train, X_test)
    else:
        raise ValueError("Method must be 'logistic_regression', 'svm' or 'random_forest' ")
