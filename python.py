import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, canny
from skimage.filters import gaussian
from skimage.transform import rescale, rotate
from skimage.util import random_noise
import tensorflow as tf
import xgboost as xgb
import optuna
from functools import partial
import joblib
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
TRAIN_PKL = 'train.pkl'
TEST_PKL = 'test.pkl'
OUTPUT_DIR = 'models'
SUBMISSION_FILE = 'submission.csv'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("🔹 Found GPU(s):", gpus if gpus else "None – running on CPU")
strategy = tf.distribute.get_strategy()
print("Using strategy:", type(strategy).__name__)

# Save helper function
def save_tf_model(model, path):
    """Save a TensorFlow model in .keras format"""
    target = os.path.join(OUTPUT_DIR, path + ".keras")
    model.save(target)
    print(f"✅ Model saved to '{target}'")

# Utility function to save any model
def save_model(model, name):
    """Save any model using joblib"""
    target = os.path.join(OUTPUT_DIR, f"{name}.joblib")
    joblib.dump(model, target)
    print(f"✅ Model saved to '{target}'")

# Load and preprocess data
def load_data(filepath):
    """Load data from pickle file and preprocess"""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Convert image lists to numpy arrays
    imgs1 = np.stack([np.array(x, dtype=np.float32) for x in data["img1"]], axis=0)
    imgs2 = np.stack([np.array(x, dtype=np.float32) for x in data["img2"]], axis=0)
    
    # Normalize images to [0, 1]
    imgs1 = imgs1 / 255.0
    imgs2 = imgs2 / 255.0
    
    if "label" in data:
        # Map labels from {-1, 1} to {0, 1} for binary classification
        labels = data["label"].map({-1: 0, 1: 1}).values
    else:
        labels = None
    
    return imgs1, imgs2, labels, data["id"].values

# Enhanced Feature Extraction
def extract_features(img_array, feature_type='all'):
    """
    Extract multiple types of features from images
    
    Parameters:
    -----------
    img_array : numpy array of shape (n_samples, height, width)
        Array of grayscale images
    feature_type : str
        Type of features to extract: 'hog', 'edge', 'texture', or 'all'
        
    Returns:
    --------
    features : numpy array
        Extracted features
    """
    features_list = []
    
    if feature_type in ['hog', 'all']:
        # Multiple HOG configurations for ensemble diversity
        hog_features1 = np.stack([
            hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                orientations=9, feature_vector=True)
            for img in img_array
        ])
        
        hog_features2 = np.stack([
            hog(img, pixels_per_cell=(6, 6), cells_per_block=(3, 3), 
                orientations=12, feature_vector=True)
            for img in img_array
        ])
        
        features_list.extend([hog_features1, hog_features2])
    
    if feature_type in ['edge', 'all']:
        # Edge detection features (useful for scissors)
        edge_features = np.stack([
            canny(img, sigma=1.0).reshape(-1)
            for img in img_array
        ])
        features_list.append(edge_features)
    
    if feature_type in ['texture', 'all']:
        # Simple texture features
        def texture_features(img):
            regions = [img[:12, :12], img[:12, 12:], img[12:, :12], img[12:, 12:]]
            features = []
            for region in regions:
                features.extend([
                    np.mean(region),
                    np.std(region),
                    np.percentile(region, 10),
                    np.percentile(region, 90)
                ])
            return np.array(features)
        
        texture_feat = np.stack([texture_features(img) for img in img_array])
        features_list.append(texture_feat)
    
    # Combine all features
    if len(features_list) == 1:
        return features_list[0]
    else:
        return np.hstack(features_list)

# Advanced Data Augmentation
def create_augmentation_model():
    """Create TensorFlow data augmentation model for training"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

# Custom paired augmentation for both images
def augment_paired_images(img1, img2, label, training=True):
    """Apply same augmentation to both images in a pair"""
    if training:
        # Stack images to apply same transform
        stacked = tf.stack([img1, img2], axis=0)
        # Apply same random transformations to both
        augmentation = create_augmentation_model()
        # Apply in a way that preserves the stack
        batch_size = tf.shape(stacked)[0]
        stacked_reshaped = tf.reshape(stacked, [1, batch_size, 24, 24, 1])
        augmented = augmentation(stacked_reshaped)
        augmented = tf.reshape(augmented, [batch_size, 24, 24, 1])
        # Unstack
        img1, img2 = augmented[0], augmented[1]
    
    # Create 2-channel image
    combined = tf.concat([img1, img2], axis=-1)
    return combined, label

# Create TensorFlow datasets
def create_tf_datasets(img1_train, img2_train, y_train, 
                      img1_val, img2_val, y_val,
                      batch_size=64):
    """Create TensorFlow datasets for training and validation"""
    # Training dataset with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices(
        ((img1_train, img2_train), y_train)
    ).map(
        lambda x, y: augment_paired_images(x[0], x[1], y, training=True),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset without augmentation
    val_ds = tf.data.Dataset.from_tensor_slices(
        ((img1_val, img2_val), y_val)
    ).map(
        lambda x, y: augment_paired_images(x[0], x[1], y, training=False),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

# Model Architectures
def build_simple_cnn(input_shape=(24, 24, 2)):
    """Build a simple CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def residual_block(x, filters, stride=1):
    """Residual block for ResNet-like architecture"""
    shortcut = x
    
    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x

def build_advanced_cnn(input_shape=(24, 24, 2)):
    """Build an advanced CNN with residual blocks"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    
    # Global pooling and output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs, name="AdvancedCNN")

def build_siamese_network(input_shape=(24, 24, 1)):
    """Build a siamese network for paired image comparison"""
    # Shared convolutional feature extractor
    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)
    
    # Shared feature extractor
    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu')
    ])
    
    # Get embeddings for both inputs
    feat_a = feature_extractor(input_a)
    feat_b = feature_extractor(input_b)
    
    # Difference layer
    diff = tf.keras.layers.Subtract()([feat_a, feat_b])
    
    # Classification head
    x = tf.keras.layers.Dense(64, activation='relu')(diff)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=[input_a, input_b], outputs=output)

# Callback for learning rate scheduling
def create_lr_scheduler():
    """Create a learning rate scheduler with cosine decay"""
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

# Main training pipeline
def train_and_evaluate():
    """Main function to train and evaluate models"""
    print("Loading and preprocessing data...")
    
    # Load and split data
    imgs1, imgs2, labels, ids = load_data(TRAIN_PKL)
    
    # Create a stratified split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        np.arange(len(labels)), labels, ids,
        train_size=0.7, random_state=SEED, stratify=labels
    )
    
    # Split the remaining 30% into equal validation and test sets
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp,
        train_size=0.5, random_state=SEED, stratify=y_temp
    )
    
    # Extract actual images for each set
    img1_train, img2_train = imgs1[X_train], imgs2[X_train]
    img1_val, img2_val = imgs1[X_val], imgs2[X_val]
    img1_test, img2_test = imgs1[X_test], imgs2[X_test]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Extract features for traditional ML models
    print("Extracting features...")
    # HOG features
    X_hog_train = np.hstack([
        extract_features(img1_train, 'hog'),
        extract_features(img2_train, 'hog')
    ])
    X_hog_val = np.hstack([
        extract_features(img1_val, 'hog'),
        extract_features(img2_val, 'hog')
    ])
    X_hog_test = np.hstack([
        extract_features(img1_test, 'hog'),
        extract_features(img2_test, 'hog')
    ])
    
    # All features
    X_all_train = np.hstack([
        extract_features(img1_train, 'all'),
        extract_features(img2_train, 'all')
    ])
    X_all_val = np.hstack([
        extract_features(img1_val, 'all'),
        extract_features(img2_val, 'all')
    ])
    X_all_test = np.hstack([
        extract_features(img1_test, 'all'),
        extract_features(img2_test, 'all')
    ])
    
    # Feature scaling for traditional ML models
    scaler = StandardScaler()
    X_hog_train_scaled = scaler.fit_transform(X_hog_train)
    X_hog_val_scaled = scaler.transform(X_hog_val)
    X_hog_test_scaled = scaler.transform(X_hog_test)
    
    scaler_all = StandardScaler()
    X_all_train_scaled = scaler_all.fit_transform(X_all_train)
    X_all_val_scaled = scaler_all.transform(X_all_val)
    X_all_test_scaled = scaler_all.transform(X_all_test)
    
    # Save scalers
    save_model(scaler, "hog_scaler")
    save_model(scaler_all, "all_features_scaler")
    
    # Create TensorFlow datasets
    train_ds, val_ds = create_tf_datasets(
        img1_train, img2_train, y_train,
        img1_val, img2_val, y_val,
        batch_size=64
    )
    
    # Base model predictions
    print("Training base models...")
    base_models_val_preds = []
    base_models_test_preds = []
    base_model_names = []
    
    # 1. XGBoost with HOG Features
    print("Training XGBoost with HOG features...")
    dtrain = xgb.DMatrix(X_hog_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_hog_val_scaled, label=y_val)
    dtest = xgb.DMatrix(X_hog_test_scaled, label=y_test)
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'reg_lambda': 1.0,
        'reg_alpha': 0.5
    }
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    # Save XGBoost model
    xgb_model.save_model(os.path.join(OUTPUT_DIR, 'xgb_hog.json'))
    
    # Get predictions
    xgb_val_preds = xgb_model.predict(dval)
    xgb_test_preds = xgb_model.predict(dtest)
    
    base_models_val_preds.append(xgb_val_preds)
    base_models_test_preds.append(xgb_test_preds)
    base_model_names.append("XGBoost HOG")
    
    # 2. XGBoost with All Features
    print("Training XGBoost with all features...")
    dtrain_all = xgb.DMatrix(X_all_train_scaled, label=y_train)
    dval_all = xgb.DMatrix(X_all_val_scaled, label=y_val)
    dtest_all = xgb.DMatrix(X_all_test_scaled, label=y_test)
    
    xgb_all_model = xgb.train(
        xgb_params,
        dtrain_all,
        num_boost_round=500,
        evals=[(dtrain_all, 'train'), (dval_all, 'val')],
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    # Save XGBoost all features model
    xgb_all_model.save_model(os.path.join(OUTPUT_DIR, 'xgb_all.json'))
    
    # Get predictions
    xgb_all_val_preds = xgb_all_model.predict(dval_all)
    xgb_all_test_preds = xgb_all_model.predict(dtest_all)
    
    base_models_val_preds.append(xgb_all_val_preds)
    base_models_test_preds.append(xgb_all_test_preds)
    base_model_names.append("XGBoost All")
    
    # 3. Simple CNN models
    print("Training simple CNN models...")
    for cnn_seed in [42, 123]:
        tf.keras.utils.set_random_seed(cnn_seed)
        
        simple_cnn = build_simple_cnn()
        simple_cnn.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            create_lr_scheduler()
        ]
        
        simple_cnn.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=callbacks,
            verbose=2
        )
        
        # Save model
        save_tf_model(simple_cnn, f"simple_cnn_seed{cnn_seed}")
        
        # Get predictions
        simple_cnn_val_preds = simple_cnn.predict(val_ds).ravel()
        simple_cnn_test_preds = simple_cnn.predict(
            tf.data.Dataset.from_tensor_slices(
                ((img1_test, img2_test), y_test)
            ).map(
                lambda x, y: augment_paired_images(x[0], x[1], y, training=False),
                num_parallel_calls=tf.data.AUTOTUNE
            ).batch(64).prefetch(tf.data.AUTOTUNE)
        ).ravel()
        
        base_models_val_preds.append(simple_cnn_val_preds)
        base_models_test_preds.append(simple_cnn_test_preds)
        base_model_names.append(f"Simple CNN {cnn_seed}")
    
    # 4. Advanced CNN models
    print("Training advanced CNN models...")
    for cnn_seed in [42, 123]:
        tf.keras.utils.set_random_seed(cnn_seed)
        
        adv_cnn = build_advanced_cnn()
        adv_cnn.compile(
            optimizer=tf.keras.optimizers.Adam(5e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            create_lr_scheduler()
        ]
        
        adv_cnn.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            callbacks=callbacks,
            verbose=2
        )
        
        # Save model
        save_tf_model(adv_cnn, f"adv_cnn_seed{cnn_seed}")
        
        # Get predictions
        adv_cnn_val_preds = adv_cnn.predict(val_ds).ravel()
        adv_cnn_test_preds = adv_cnn.predict(
            tf.data.Dataset.from_tensor_slices(
                ((img1_test, img2_test), y_test)
            ).map(
                lambda x, y: augment_paired_images(x[0], x[1], y, training=False),
                num_parallel_calls=tf.data.AUTOTUNE
            ).batch(64).prefetch(tf.data.AUTOTUNE)
        ).ravel()
        
        base_models_val_preds.append(adv_cnn_val_preds)
        base_models_test_preds.append(adv_cnn_test_preds)
        base_model_names.append(f"Advanced CNN {cnn_seed}")
    
    # 5. Siamese Network
    print("Training siamese network...")
    # Reshape for siamese network
    img1_train_reshaped = np.expand_dims(img1_train, -1)
    img2_train_reshaped = np.expand_dims(img2_train, -1)
    img1_val_reshaped = np.expand_dims(img1_val, -1)
    img2_val_reshaped = np.expand_dims(img2_val, -1)
    img1_test_reshaped = np.expand_dims(img1_test, -1)
    img2_test_reshaped = np.expand_dims(img2_test, -1)
    
    siamese_net = build_siamese_network()
    siamese_net.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create siamese datasets
    siamese_train_ds = tf.data.Dataset.from_tensor_slices(
        ((img1_train_reshaped, img2_train_reshaped), y_train)
    ).batch(64).prefetch(tf.data.AUTOTUNE)
    
    siamese_val_ds = tf.data.Dataset.from_tensor_slices(
        ((img1_val_reshaped, img2_val_reshaped), y_val)
    ).batch(64).prefetch(tf.data.AUTOTUNE)
    
    # Train
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        ),
        create_lr_scheduler()
    ]
    
    siamese_net.fit(
        siamese_train_ds,
        validation_data=siamese_val_ds,
        epochs=100,
        callbacks=callbacks,
        verbose=2
    )
    
    # Save model
    save_tf_model(siamese_net, "siamese_net")
    
    # Get predictions
    siamese_val_preds = siamese_net.predict(siamese_val_ds).ravel()
    
    siamese_test_ds = tf.data.Dataset.from_tensor_slices(
        ((img1_test_reshaped, img2_test_reshaped), y_test)
    ).batch(64).prefetch(tf.data.AUTOTUNE)
    
    siamese_test_preds = siamese_net.predict(siamese_test_ds).ravel()
    
    base_models_val_preds.append(siamese_val_preds)
    base_models_test_preds.append(siamese_test_preds)
    base_model_names.append("Siamese Network")
    
    # 6. Random Forest on all features
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        random_state=SEED,
        n_jobs=-1
    )
    rf_model.fit(X_all_train_scaled, y_train)
    
    # Save model
    save_model(rf_model, "random_forest")
    
    # Get predictions
    rf_val_preds = rf_model.predict_proba(X_all_val_scaled)[:, 1]
    rf_test_preds = rf_model.predict_proba(X_all_test_scaled)[:, 1]
    
    base_models_val_preds.append(rf_val_preds)
    base_models_test_preds.append(rf_test_preds)
    base_model_names.append("Random Forest")
    
    # Stack predictions for meta-learning
    print("Training meta-learner...")
    meta_X_val = np.column_stack(base_models_val_preds)
    meta_X_test = np.column_stack(base_models_test_preds)
    
    # Use Gradient Boosting as meta-learner
    meta_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=SEED
    )
    meta_model.fit(meta_X_val, y_val)
    
    # Save meta-model
    save_model(meta_model, "meta_learner")
    
    # Evaluate on test set
    meta_test_preds = meta_model.predict_proba(meta_X_test)[:, 1]
    meta_test_binary = (meta_test_preds > 0.5).astype(int)
    meta_test_accuracy = accuracy_score(y_test, meta_test_binary)
    
    print(f"Meta-learner test accuracy: {meta_test_accuracy:.4f}")
    
    # Base model performances
    print("\nBase model performances on validation set:")
    for i, name in enumerate(base_model_names):
        binary_preds = (base_models_val_preds[i] > 0.5).astype(int)
        acc = accuracy_score(y_val, binary_preds)
        print(f"{name}: {acc:.4f}")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, meta_test_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    
    # Feature importance for meta-learner
    plt.figure(figsize=(10, 6))
    importances = meta_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [base_model_names[i] for i in indices], rotation=45)
    plt.title('Meta-learner Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'meta_feature_importance.png'))
    
    return {
        'xgb_hog': xgb_model,
        'xgb_all': xgb_all_model,
        'scalers': {
            'hog': scaler,
            'all': scaler_all
        },
        'simple_cnns': [f"simple_cnn_seed{seed}" for seed in [42, 123]],
        'adv_cnns': [f"adv_cnn_seed{seed}" for seed in [42, 123]],
        'siamese': "siamese_net",
        'meta_model': meta_model
    }