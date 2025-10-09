# medical_image_classifier.py
# Professional Tuberculosis & Fungal Disease Classification System
# Complete standalone code - Save and run directly

import os
import json
import warnings

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

# Configuration
warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

print("🏥 PROFESSIONAL MEDICAL IMAGE CLASSIFICATION SYSTEM")
print("=" * 65)
print("🔬 Tuberculosis vs Fungal Disease vs Normal Classification")
print("=" * 65)


class MedicalImageClassifier:
    def __init__(self,
                 img_size=(224, 224),
                 batch_size=16,
                 num_classes=3,
                 class_names=None,
                 base_model_weights='imagenet'):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_names = class_names or ['tuberculosis', 'fungal', 'normal']
        self.model = None
        self.history = None
        self.base_model_weights = base_model_weights

    def setup_environment(self):
        """Create directory structure and verify dependencies"""
        print("\n📋 SETTING UP ENVIRONMENT...")

        # Check if TensorFlow is available
        try:
            print(f"✅ TensorFlow Version: {tf.__version__}")
        except Exception:
            print("❌ TensorFlow not installed. Please run: pip install tensorflow")
            return False

        # Create directory structure
        directories = [
            'dataset/train/tuberculosis',
            'dataset/train/fungal',
            'dataset/train/normal',
            'dataset/val/tuberculosis',
            'dataset/val/fungal',
            'dataset/val/normal',
            'dataset/test/tuberculosis',
            'dataset/test/fungal',
            'dataset/test/normal',
            'models',
            'results',
            'logs'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Create sample dataset structure (only if directories are empty)
        self.create_sample_dataset()

        print("✅ Environment setup completed successfully!")
        return True

    def create_sample_dataset(self):
        """Create sample dataset structure with placeholder images (only when empty)"""
        print("\n📁 CREATING DATASET STRUCTURE (if empty)...")

        # Create sample images for demonstration if directories are empty
        for split in ['train', 'val', 'test']:
            for class_idx, class_name in enumerate(self.class_names):
                dir_path = f'dataset/{split}/{class_name}'
                os.makedirs(dir_path, exist_ok=True)

                # Create placeholder images if directory is empty
                existing_imgs = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if len(existing_imgs) == 0:
                    # Create different colored images for each class
                    if class_name == 'tuberculosis':
                        color = (100, 100, 200)  # Blue-ish
                    elif class_name == 'fungal':
                        color = (100, 200, 100)  # Green-ish
                    else:  # normal
                        color = (200, 100, 100)  # Red-ish

                    # Create multiple sample images
                    for i in range(5):
                        img = np.ones((self.img_size[0], self.img_size[1], 3), dtype=np.uint8) * 50
                        img[:, :, 0] = np.clip(color[0] + i * 10, 0, 255)
                        img[:, :, 1] = np.clip(color[1] + i * 10, 0, 255)
                        img[:, :, 2] = np.clip(color[2] + i * 10, 0, 255)

                        # Add some variation
                        cv2.putText(img, f'{class_name}_{i}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                        cv2.imwrite(f'{dir_path}/sample_{i}.jpg', img)

        print("💡 DATASET STRUCTURE READY!")
        print("   Place your actual medical images in the 'dataset/' subfolders if you want real training.")

    def create_hybrid_model(self):
        """Create advanced hybrid CNN model with custom and transfer learning"""
        print("\n🧠 BUILDING HYBRID CNN MODEL...")
        try:
            # Input layer
            inputs = layers.Input(shape=(*self.img_size, 3))

            # === CUSTOM CNN BRANCH ===
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.2)(x)

            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.3)(x)

            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.3)(x)

            # === TRANSFER LEARNING BRANCH ===
            try:
                base_model = EfficientNetB0(
                    weights=self.base_model_weights,
                    include_top=False,
                    input_tensor=inputs
                )
                print("✅ Loaded EfficientNetB0 with ImageNet weights")
            except Exception as e:
                print(f"⚠️ Failed to load ImageNet weights: {e}")
                base_model = EfficientNetB0(
                    weights=None,
                    include_top=False,
                    input_tensor=inputs
                )
                print("⚠️ Loaded EfficientNetB0 WITHOUT pretrained weights")

            base_model.trainable = False

            y = base_model.output
            y = layers.GlobalAveragePooling2D()(y)
            y = layers.Dense(128, activation='relu')(y)
            y = layers.BatchNormalization()(y)
            y = layers.Dropout(0.4)(y)

            # === COMBINE BRANCHES ===
            custom_features = layers.Flatten()(x)
            combined = layers.concatenate([custom_features, y])

            # === CLASSIFICATION HEAD ===
            combined = layers.Dense(256, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(0.5)(combined)

            combined = layers.Dense(128, activation='relu')(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Dropout(0.3)(combined)

            # Output layer
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(combined)

            model = Model(inputs=inputs, outputs=outputs)
            print("✅ Hybrid CNN model created successfully!")
            return model

        except Exception as e:
            print(f"❌ Error during model building setup: {e}")
            return None

    def prepare_data(self):
        """Prepare data generators with professional augmentation"""
        print("\n📊 PREPARING DATA GENERATORS...")

        try:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            # Validation and test (only rescaling)
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            # Create generators
            train_generator = train_datagen.flow_from_directory(
                'dataset/train',
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=True
            )

            val_generator = test_datagen.flow_from_directory(
                'dataset/val',
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )

            test_generator = test_datagen.flow_from_directory(
                'dataset/test',
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )

            print("✅ Data generators prepared successfully!")
            print(f"📈 Training samples: {train_generator.samples}")
            print(f"📈 Validation samples: {val_generator.samples}")
            print(f"📈 Test samples: {test_generator.samples}")
            print(f"🏷️  Class mapping: {train_generator.class_indices}")

            return train_generator, val_generator, test_generator

        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return None, None, None

    def train_model(self, train_gen, val_gen, epochs=15):
        """Train the model with professional configuration"""
        print("\n🎯 STARTING MODEL TRAINING...")

        try:
            if self.model is None:
                print("❌ No model found. Please build the model first.")
                return None

            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Professional callbacks
            callbacks = [
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]

            # Train model
            print("🔄 Training in progress... This may take a few minutes.")
            history = self.model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )

            print("✅ Model training completed successfully!")
            return history

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return None

    def visualize_training(self, history):
        """Create professional training visualizations"""
        print("\n📈 GENERATING TRAINING VISUALIZATIONS...")

        try:
            plt.style.use('seaborn-v0_8')
            # Ensure history exists
            if history is None:
                print("⚠️ No history available to visualize.")
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Accuracy plot
            ax1.plot(history.history.get('accuracy', []), linewidth=2, label='Training Accuracy')
            ax1.plot(history.history.get('val_accuracy', []), linewidth=2, label='Validation Accuracy')
            ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Loss plot
            ax2.plot(history.history.get('loss', []), linewidth=2, label='Training Loss')
            ax2.plot(history.history.get('val_loss', []), linewidth=2, label='Validation Loss')
            ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ Training visualizations saved!")

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")

    def evaluate_model(self, test_gen):
        """Comprehensive model evaluation"""
        print("\n📊 EVALUATING MODEL PERFORMANCE...")

        try:
            if self.model is None:
                print("❌ No model available for evaluation.")
                return 0.0

            # Load best model if available
            if os.path.exists('models/best_model.h5'):
                try:
                    self.model.load_weights('models/best_model.h5')
                    print("✅ Loaded best model weights")
                except Exception as e:
                    print(f"⚠️ Failed to load best_model.h5: {e}")

            # Evaluate
            test_loss, test_accuracy = self.model.evaluate(test_gen, verbose=0)

            print("\n" + "=" * 50)
            print("🎯 MODEL PERFORMANCE SUMMARY")
            print("=" * 50)
            print(f"📉 Test Loss: {test_loss:.4f}")
            print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
            print("=" * 50)

            # Predictions
            predictions = self.model.predict(test_gen, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = test_gen.classes
            class_labels = list(test_gen.class_indices.keys())

            # Classification report
            print("\n📋 DETAILED CLASSIFICATION REPORT:")
            print("=" * 50)
            report = classification_report(true_classes, predicted_classes,
                                           target_names=class_labels, digits=4)
            print(report)

            # Confusion matrix
            self.plot_confusion_matrix(true_classes, predicted_classes, class_labels)

            # Save results
            self.save_results(test_accuracy, test_loss, class_labels)

            return test_accuracy

        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return 0.0

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Create professional confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        cbar_kws={'shrink': 0.8})
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontweight='bold')
            plt.ylabel('True Label', fontweight='bold')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ Confusion matrix saved!")

        except Exception as e:
            print(f"❌ Error creating confusion matrix: {e}")

    def save_results(self, accuracy, loss, class_labels):
        """Save results to JSON file"""
        try:
            results = {
                'test_accuracy': float(accuracy),
                'test_loss': float(loss),
                'accuracy_percentage': float(accuracy * 100),
                'model_architecture': 'Hybrid CNN (Custom + EfficientNetB0)',
                'classes': class_labels,
                'image_size': list(self.img_size),
                'batch_size': self.batch_size
            }

            with open('results/model_performance.json', 'w') as f:
                json.dump(results, f, indent=4)

            print("💾 Results saved to 'results/model_performance.json'")

        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def run_complete_pipeline(self):
        """Execute complete classification pipeline"""
        print("🚀 STARTING COMPLETE MEDICAL IMAGE CLASSIFICATION PIPELINE")
        print("=" * 70)

        try:
            # Step 1: Setup environment
            if not self.setup_environment():
                return

            # Step 2: Prepare data
            train_gen, val_gen, test_gen = self.prepare_data()
            if train_gen is None:
                print("❌ Failed to prepare data. Please check your dataset.")
                return

            # Step 3: Create model
            self.model = self.create_hybrid_model()
            if self.model is None:
                print("❌ Failed to create model.")
                return

            # Display model architecture
            print("\n🏗️ MODEL ARCHITECTURE SUMMARY:")
            print("=" * 50)
            self.model.summary()

            # Step 4: Train model
            self.history = self.train_model(train_gen, val_gen, epochs=15)
            if self.history is None:
                print("❌ Training failed.")
                return

            # Step 5: Visualize training
            self.visualize_training(self.history)

            # Step 6: Evaluate model
            final_accuracy = self.evaluate_model(test_gen)

            # Step 7: Save final model
            try:
                self.model.save('models/final_medical_model.h5')
                print(f"\n💾 Final model saved as 'models/final_medical_model.h5'")
            except Exception as e:
                print(f"⚠️ Could not save final model: {e}")

            # Final report
            print("\n" + "🎉" * 20)
            print("🚀 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"🎯 FINAL TEST ACCURACY: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
            print("\n📁 GENERATED FILES:")
            print("   - models/best_model.h5 (Best model during training)")
            print("   - models/final_medical_model.h5 (Final trained model)")
            print("   - results/training_history.png (Training plots)")
            print("   - results/confusion_matrix.png (Confusion matrix)")
            print("   - results/model_performance.json (Performance metrics)")
            print("🎉" * 20)

        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("   1. Make sure all required packages are installed")
            print("   2. Check that you have images in dataset folders")
            print("   3. Verify image formats (JPEG, PNG supported)")
            print("   4. Ensure sufficient disk space")


def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 CHECKING DEPENDENCIES...")

    required_packages = {
        'tensorflow': 'TensorFlow',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'numpy': 'NumPy'
    }

    missing_packages = []

    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2  # noqa: F401
            elif package == 'sklearn':
                import sklearn  # noqa: F401
            else:
                __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            missing_packages.append(name)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Install them using: pip install " + " ".join(missing_packages))
        return False

    print("✅ All dependencies available!")
    return True


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("🏥 PROFESSIONAL MEDICAL IMAGE CLASSIFIER")
    print("🔬 Tuberculosis vs Fungal vs Normal Classification")
    print("=" * 70)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and run again.")
        return

    # Create classifier instance
    classifier = MedicalImageClassifier(img_size=(224, 224), batch_size=16, num_classes=3)

    # Run full pipeline
    classifier.run_complete_pipeline()


if __name__ == "__main__":
    main()
