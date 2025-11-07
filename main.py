# train_and_save_models.py (CORRECTED VERSION)
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FEATURE EXTRACTION AND MODEL CLASSES
# =============================================================================

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, duration=2, hop_length=512):
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = hop_length
        self.scaler = StandardScaler()
        
    def extract_features(self, file_path):
        """Extract audio features from a single file"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            features = []
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # RMS energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
            features.extend(np.mean(contrast, axis=1))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            features.extend(np.mean(chroma, axis=1))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def prepare_dataset(self, data_directory):
        """Prepare dataset from directory containing audio files"""
        features = []
        labels = []
        file_names = []
        
        # Define class mapping based on file naming convention
        class_mapping = {
            'idle': 'idle',
            'normal': 'normal_flow', 
            'leakage': 'leakage'
        }
        
        # Check if directory exists
        if not os.path.exists(data_directory):
            print(f"ERROR: Data directory '{data_directory}' does not exist.")
            return np.array([]), np.array([]), []
        
        # Get all audio files
        audio_files = [f for f in os.listdir(data_directory) 
                      if f.lower().endswith(('.wav', '.s3.wav'))]
        
        if not audio_files:
            print(f"ERROR: No audio files found in {data_directory}")
            return np.array([]), np.array([]), []
        
        print(f"Found {len(audio_files)} audio files")
        
        for filename in audio_files:
            file_path = os.path.join(data_directory, filename)
            
            # Determine label based on filename
            label = None
            for key, value in class_mapping.items():
                if key in filename.lower():
                    label = value
                    break
            
            if label:
                print(f"Processing: {filename} -> {label}")
                feature_vector = self.extract_features(file_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(label)
                    file_names.append(filename)
                else:
                    print(f"WARNING: Could not extract features from {filename}")
            else:
                print(f"WARNING: Could not determine label for {filename}")
        
        return np.array(features), np.array(labels), file_names

class WaterFlowClassifier:
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = [
            'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5',
            'mfcc_mean_6', 'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9', 'mfcc_mean_10',
            'mfcc_mean_11', 'mfcc_mean_12', 'mfcc_mean_13', 'mfcc_std_1', 'mfcc_std_2',
            'mfcc_std_3', 'mfcc_std_4', 'mfcc_std_5', 'mfcc_std_6', 'mfcc_std_7',
            'mfcc_std_8', 'mfcc_std_9', 'mfcc_std_10', 'mfcc_std_11', 'mfcc_std_12',
            'mfcc_std_13', 'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std', 'zcr_mean', 'zcr_std',
            'rms_mean', 'rms_std', 'spectral_contrast_1', 'spectral_contrast_2',
            'spectral_contrast_3', 'spectral_contrast_4', 'spectral_contrast_5',
            'spectral_contrast_6', 'spectral_contrast_7', 'chroma_1', 'chroma_2',
            'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8',
            'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12'
        ]
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y_encoded)
        self.models['random_forest'] = rf
        
        # SVM
        print("Training SVM...")
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_scaled, y_encoded)
        self.models['svm'] = svm
        
        return X_scaled, y_encoded
    
    def predict(self, X, model_name='random_forest'):
        """Make predictions using specified model"""
        X_scaled = self.scaler.transform(X)
        model = self.models[model_name]
        predictions_encoded = model.predict(X_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions
    
    def predict_proba(self, X, model_name='random_forest'):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        model = self.models[model_name]
        return model.predict_proba(X_scaled)
    
    def evaluate_model(self, X_test, y_test, model_name='random_forest'):
        """Evaluate model performance"""
        predictions = self.predict(X_test, model_name)
        
        print(f"Model: {model_name}")
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return predictions

# =============================================================================
# MODEL SAVING FUNCTIONS
# =============================================================================

def save_trained_models(ml_classifier, feature_extractor, save_dir="saved_models"):
    """
    Save all trained models and components to disk
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save ML classifier components
    model_data = {
        'random_forest': ml_classifier.models['random_forest'],
        'svm': ml_classifier.models['svm'],
        'label_encoder': ml_classifier.label_encoder,
        'scaler': ml_classifier.scaler,
        'feature_names': ml_classifier.feature_names
    }
    
    # Save using joblib (recommended for scikit-learn models)
    joblib.dump(model_data, os.path.join(save_dir, 'ml_classifier.joblib'))
    print(f"SUCCESS: ML classifier saved to {save_dir}/ml_classifier.joblib")
    
    # Save feature extractor configuration
    feature_extractor_config = {
        'sample_rate': feature_extractor.sample_rate,
        'duration': feature_extractor.duration,
        'hop_length': feature_extractor.hop_length
    }
    joblib.dump(feature_extractor_config, os.path.join(save_dir, 'feature_extractor_config.joblib'))
    print(f"SUCCESS: Feature extractor config saved to {save_dir}/feature_extractor_config.joblib")
    
    # Save complete model info
    model_info = {
        'saved_at': np.datetime64('now'),
        'ml_models': ['random_forest', 'svm'],
        'feature_extractor_params': feature_extractor_config,
        'classes': list(ml_classifier.label_encoder.classes_)
    }
    joblib.dump(model_info, os.path.join(save_dir, 'model_info.joblib'))
    print(f"SUCCESS: Model info saved to {save_dir}/model_info.joblib")
    
    print(f"ALL MODELS: All models saved successfully in '{save_dir}' directory!")
    return save_dir

# =============================================================================
# MAIN TRAINING FUNCTION (CORRECTED)
# =============================================================================

def train_models(data_directory=None):
    """
    Main function to train water flow classification models
    """
    if data_directory is None:
        # Try to find the data directory
        possible_paths = [
            r"C:\a_python\accoustics\train",
            r"C:\a_python\accoustics\data",
            r"./train",
            r"./data"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_directory = path
                break
        
        if data_directory is None:
            print("ERROR: Could not find data directory. Please specify the path.")
            return None, None
    
    print("=" * 60)
    print("WATER FLOW AUDIO CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    print(f"Using data directory: {data_directory}")
    
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor()
    
    # Extract features
    print("\nStep 1: Extracting features from audio files...")
    features, labels, file_names = feature_extractor.prepare_dataset(data_directory)
    
    if len(features) == 0:
        print("ERROR: No features extracted. Please check:")
        print("1. Your data directory path is correct")
        print("2. Audio files are in WAV format")
        print("3. Files follow the naming convention (idle, normal, leakage)")
        return None, None
    
    print(f"SUCCESS: Extracted features from {len(features)} files")
    
    # FIXED LINE: Correct f-string syntax
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique_labels, counts))
    print(f"Class distribution: {class_distribution}")
    
    # Split the data
    print("\nStep 2: Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train traditional ML models
    print("\nStep 3: Training machine learning models...")
    ml_classifier = WaterFlowClassifier()
    X_train_scaled, y_train_encoded = ml_classifier.train_models(X_train, y_train)
    
    # Evaluate models
    print("\nStep 4: Evaluating model performance...")
    
    # Evaluate Random Forest
    print("\nEvaluating Random Forest...")
    rf_predictions = ml_classifier.evaluate_model(X_test, y_test, 'random_forest')
    
    # Evaluate SVM
    print("\nEvaluating SVM...")
    svm_predictions = ml_classifier.evaluate_model(X_test, y_test, 'svm')
    
    # Feature importance for Random Forest
    print("\nStep 5: Analyzing feature importance...")
    rf_model = ml_classifier.models['random_forest']
    feature_importance = rf_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), feature_importance[top_features_idx])
    plt.yticks(range(10), [ml_classifier.feature_names[i] for i in top_features_idx])
    plt.title('Top 10 Most Important Features (Random Forest)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return ml_classifier, feature_extractor

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_new_audio(file_path, ml_classifier, feature_extractor, model_name='random_forest'):
    """Predict class for a new audio file"""
    # Extract features
    features = feature_extractor.extract_features(file_path)
    
    if features is not None:
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = ml_classifier.predict(features, model_name)[0]
        probabilities = ml_classifier.predict_proba(features, model_name)[0]
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"Predicted class: {prediction}")
        print("Probabilities:")
        for class_name, prob in zip(ml_classifier.label_encoder.classes_, probabilities):
            print(f"  {class_name}: {prob:.4f}")
        
        return prediction, probabilities
    else:
        print(f"ERROR: Could not extract features from {file_path}")
        return None, None

# =============================================================================
# LOAD MODELS FUNCTION (as requested)
# =============================================================================

def load_models(models_dir="saved_models"):
    """Load saved models"""
    try:
        model_data = joblib.load(os.path.join(models_dir, 'ml_classifier.joblib'))
        feature_config = joblib.load(os.path.join(models_dir, 'feature_extractor_config.joblib'))
        
        # Recreate classifier object
        class LoadedWaterFlowClassifier:
            def __init__(self, model_data):
                self.models = {
                    'random_forest': model_data['random_forest'],
                    'svm': model_data['svm']
                }
                self.label_encoder = model_data['label_encoder']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
            
            def predict(self, X, model_name='random_forest'):
                X_scaled = self.scaler.transform(X)
                model = self.models[model_name]
                predictions_encoded = model.predict(X_scaled)
                predictions = self.label_encoder.inverse_transform(predictions_encoded)
                return predictions
            
            def predict_proba(self, X, model_name='random_forest'):
                X_scaled = self.scaler.transform(X)
                model = self.models[model_name]
                return model.predict_proba(X_scaled)
        
        # Recreate feature extractor
        class LoadedAudioFeatureExtractor:
            def __init__(self, config):
                self.sample_rate = config['sample_rate']
                self.duration = config['duration']
                self.hop_length = config['hop_length']
                self.scaler = StandardScaler()
            
            def extract_features(self, file_path):
                """Extract audio features from a single file"""
                try:
                    # Load audio file
                    y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
                    
                    features = []
                    
                    # MFCC features
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
                    features.extend(np.mean(mfcc, axis=1))
                    features.extend(np.std(mfcc, axis=1))
                    
                    # Spectral features
                    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
                    features.append(np.mean(spectral_centroid))
                    features.append(np.std(spectral_centroid))
                    
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
                    features.append(np.mean(spectral_rolloff))
                    features.append(np.std(spectral_rolloff))
                    
                    # Zero crossing rate
                    zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
                    features.append(np.mean(zcr))
                    features.append(np.std(zcr))
                    
                    # RMS energy
                    rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
                    features.append(np.mean(rms))
                    features.append(np.std(rms))
                    
                    # Spectral contrast
                    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
                    features.extend(np.mean(contrast, axis=1))
                    
                    # Chroma features
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
                    features.extend(np.mean(chroma, axis=1))
                    
                    return np.array(features)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    return None
        
        ml_classifier = LoadedWaterFlowClassifier(model_data)
        feature_extractor = LoadedAudioFeatureExtractor(feature_config)
        
        print("SUCCESS: Models loaded successfully!")
        return ml_classifier, feature_extractor
        
    except Exception as e:
        print(f"ERROR: Could not load models: {e}")
        return None, None

# =============================================================================
# COMPLETE WORKFLOW FUNCTION
# =============================================================================

def complete_workflow():
    """
    Complete workflow: Train models, save them, and test predictions
    """
    print("STARTING COMPLETE WORKFLOW")
    print("=" * 50)
    
    # Step 1: Train models
    print("\n1. TRAINING MODELS...")
    ml_classifier, feature_extractor = train_models()
    
    if ml_classifier is None:
        print("ERROR: Training failed. Cannot proceed.")
        return
    
    # Step 2: Save models
    print("\n2. SAVING MODELS...")
    save_dir = save_trained_models(ml_classifier, feature_extractor)
    
    # Step 3: Test prediction with saved models
    print("\n3. TESTING PREDICTION WITH SAVED MODELS...")
    test_file = r"C:\a_python\accoustics\test\water flow - leakage.364dm39i.s6.wav"
    
    if os.path.exists(test_file):
        print(f"Testing prediction on: {os.path.basename(test_file)}")
        
        # Load models and predict
        loaded_ml_classifier, loaded_feature_extractor = load_models(save_dir)
        
        if loaded_ml_classifier is not None:
            prediction, probabilities = predict_new_audio(
                test_file, loaded_ml_classifier, loaded_feature_extractor
            )
            
            if prediction:
                print(f"SUCCESS: Prediction test completed!")
                print(f"Final prediction: {prediction}")
                print(f"Confidence: {max(probabilities):.2%}")
        else:
            print("ERROR: Could not load models for testing")
    else:
        print(f"WARNING: Test file not found: {test_file}")
        print("You can test predictions later using the saved models.")
    
    # Step 4: Create prediction script
    print("\n4. CREATING PREDICTION SCRIPT...")
    create_prediction_script()
    
    print("\n" + "=" * 50)
    print("WORKFLOW COMPLETE!")
    print("=" * 50)
    print("\nYou can now use the saved models for predictions:")
    print("\nOption 1: Use the load_models function:")
    print('''
from train_and_save_models import load_models, predict_new_audio

# Load models
ml_classifier, feature_extractor = load_models("saved_models")

# Make prediction
prediction, probabilities = predict_new_audio(
    "path/to/audio.wav", 
    ml_classifier, 
    feature_extractor
)
''')
    
    print("\nOption 2: Use the standalone predictor:")
    print('python standalone_predictor.py "path/to/audio.wav"')

def create_prediction_script():
    """
    Create a standalone prediction script
    """
    script_content = '''# standalone_predictor.py
# Water Flow Audio Classification - Standalone Predictor
# Usage: python standalone_predictor.py "path/to/audio/file.wav"

import os
import sys
import joblib
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder

class StandaloneWaterFlowPredictor:
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load ML classifier
            model_data = joblib.load(os.path.join(self.models_dir, 'ml_classifier.joblib'))
            self.models = model_data
            self.models_loaded = True
            print("SUCCESS: Models loaded successfully")
            return True
        except Exception as e:
            print(f"ERROR: Error loading models: {e}")
            return False
    
    def extract_features(self, file_path, sample_rate=22050, duration=2, hop_length=512):
        """Extract audio features"""
        try:
            y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
            
            features = []
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # RMS energy
            rms = librosa.feature.rms(y=y, hop_length=hop_length)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
            features.extend(np.mean(contrast, axis=1))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            features.extend(np.mean(chroma, axis=1))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def predict(self, file_path, model_name='random_forest'):
        """Make prediction for audio file"""
        if not self.models_loaded:
            print("ERROR: Models not loaded")
            return None, None
            
        features = self.extract_features(file_path)
        
        if features is not None:
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.models['scaler'].transform(features)
            
            # Get prediction
            model = self.models[model_name]
            prediction_encoded = model.predict(features_scaled)[0]
            prediction = self.models['label_encoder'].inverse_transform([prediction_encoded])[0]
            
            # Get probabilities
            probabilities = model.predict_proba(features_scaled)[0]
            
            return prediction, probabilities
        else:
            return None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python standalone_predictor.py <audio_file_path> [model_name]")
        print("Available models: random_forest, svm")
        return
    
    audio_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'random_forest'
    
    if not os.path.exists(audio_file):
        print(f"ERROR: File not found: {audio_file}")
        return
    
    predictor = StandaloneWaterFlowPredictor()
    
    if predictor.models_loaded:
        prediction, probabilities = predictor.predict(audio_file, model_name)
        
        if prediction:
            print("PREDICTION RESULTS")
            print(f"File: {os.path.basename(audio_file)}")
            print(f"Prediction: {prediction.upper()}")
            print("Probabilities:")
            for i, prob in enumerate(probabilities):
                class_name = predictor.models['label_encoder'].classes_[i]
                print(f"   {class_name}: {prob:.4f}")
            
            max_prob = max(probabilities)
            confidence = "HIGH" if max_prob > 0.8 else "MEDIUM" if max_prob > 0.6 else "LOW"
            print(f"Confidence: {max_prob:.2%} ({confidence})")
        else:
            print("ERROR: Could not process audio file")
    else:
        print("ERROR: Could not load models")

if __name__ == "__main__":
    main()
'''
    
    with open("standalone_predictor.py", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print("SUCCESS: Standalone predictor script created: standalone_predictor.py")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("WATER FLOW AUDIO CLASSIFICATION - TRAIN AND SAVE MODELS")
    print("This script will:")
    print("1. Train ML models on your audio data")
    print("2. Save the trained models as joblib files")
    print("3. Create a standalone prediction script")
    print("4. Test the prediction system with saved models")
    print("=" * 60)
    
    # Run complete workflow
    complete_workflow()