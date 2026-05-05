import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    """
    A robust pipeline for training a MobileNetV2 model on Log-Mel Spectrograms 
    for Selective Active Noise Cancellation (ANC).
    """
    
    def __init__(self, data_dir: Path, model_save_dir: Path, batch_size: int = 32, epochs: int = 50):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = 0
        self.label_encoder = LabelEncoder()
        
        # Ensure save directory exists
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads .npy files, encodes labels, duplicates channels for RGB emulation, 
        and splits the dataset into train, validation, and test sets.
        """
        logging.info("Loading processed features and labels...")
        try:
            X_raw = np.load(self.data_dir / "X_features.npy")
            y_raw = np.load(self.data_dir / "y_labels.npy")
        except FileNotFoundError as e:
            logging.error(f"Data files not found. Ensure preprocessing is complete. Details: {e}")
            raise

        # Encode string labels (e.g., 'siren', 'dog') to integers
        y_encoded = self.label_encoder.fit_transform(y_raw)
        self.num_classes = len(self.label_encoder.classes_)
        logging.info(f"Detected {self.num_classes} distinct classes: {self.label_encoder.classes_}")
        
        # Convert integers to One-Hot Encoding for Categorical Cross-Entropy
        y_categorical = to_categorical(y_encoded, num_classes=self.num_classes)

        # Reshape X to include the channel dimension (Samples, Mel_Bins, Time_Steps, 1)
        X_expanded = np.expand_dims(X_raw, axis=-1)
        
        # Duplicate the single grayscale channel 3 times to satisfy MobileNetV2's RGB requirement
        X_rgb_emulated = np.repeat(X_expanded, 3, axis=-1)
        logging.info(f"Reshaped input data for MobileNetV2. New shape: {X_rgb_emulated.shape}")

        # Split: 80% Train, 10% Validation, 10% Test
        X_temp, X_test, y_temp, y_test = train_test_split(X_rgb_emulated, y_categorical, test_size=0.10, random_state=42, stratify=y_categorical)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp) # 0.1111 of 0.90 is ~0.10

        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """
        Builds the MobileNetV2 architecture with Transfer Learning applied.
        """
        logging.info("Building MobileNetV2 architecture...")
        
        # Define Input
        inputs = Input(shape=input_shape)
        
        # Load pre-trained MobileNetV2 without the top classification layer
        base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
        
        # Freeze the base model to prevent destroying pre-trained weights during early training
        base_model.trainable = False
        
        # Add custom top layers for our specific classification task
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary(print_fn=logging.info)
        return model

    def train(self) -> Optional[Model]:
        """
        Executes the full training pipeline including data preparation, model building, and evaluation.
        """
        # 1. Prepare Data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_prepare_data()
        input_shape = X_train.shape[1:]

        # 2. Build Model
        model = self.build_model(input_shape)

        # 3. Define Callbacks (Early Stopping & Model Checkpoint)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=str(self.model_save_dir / "best_mobilenetv2_anc.h5"), 
                            monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]

        # 4. Train
        logging.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # 5. Evaluate on unseen Test Data
        logging.info("Evaluating model on Test Set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

        return model

# --- Execution ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_SAVE_DIR = BASE_DIR / "saved_models" / "base_models"
    
    trainer = ModelTrainer(
        data_dir=PROCESSED_DATA_DIR,
        model_save_dir=MODEL_SAVE_DIR,
        batch_size=32,
        epochs=50 # Early stopping will likely halt this earlier
    )
    
    trained_model = trainer.train()