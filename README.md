# step 1 - run main.py file to generate joblib file and create machine learning model
## step 2 - run realtimegenerator.py to see the results from soundwaves file
This code implements a complete water flow audio classification system that trains machine learning models to classify audio into three categories: idle, normal flow, and leakage.

Main Components:
1. Audio Feature Extraction (AudioFeatureExtractor)
Extracts 52 audio features from WAV files including:

MFCCs (mean and standard deviation)

Spectral features (centroid, rolloff)

Zero crossing rate

RMS energy

Spectral contrast

Chroma features

2. Machine Learning Classifier (WaterFlowClassifier)
Trains and manages two ML models:

Random Forest (100 estimators)

Support Vector Machine (SVM with RBF kernel)

Includes label encoding and feature scaling

Provides prediction and evaluation methods

3. Model Training Pipeline (train_models)
Automatically finds audio data directory

Extracts features from training data

Splits data into train/test sets (80/20)

Trains both ML models

Evaluates performance with classification reports and confusion matrices

Displays feature importance analysis

4. Model Persistence
Saves trained models, scalers, and configuration as joblib files

Loads saved models for future predictions

Creates a complete model package in saved_models/ directory

5. Prediction System
predict_new_audio(): Classifies new audio files

load_models(): Reloads saved models for inference

Creates a standalone predictor script for deployment

Key Features:
Automatic data discovery - searches for audio files in common directories

Comprehensive evaluation - confusion matrices, classification reports, feature importance

Production-ready - saves everything needed for deployment

Error handling - robust file processing with error messages

Visualization - plots confusion matrices and feature importance

Usage:
The system can be used to classify water flow sounds as:

Idle - no water flow

Normal flow - regular water flow

Leakage - potential leak detection

The code provides both training capabilities and a ready-to-use prediction interface for real-world deployment.

markdown file
markdown
# Water Flow Audio Classification System

A complete machine learning system for classifying water flow audio signals into three categories: **idle**, **normal flow**, and **leakage**.

## Overview

This system uses audio feature extraction and machine learning to automatically classify water flow patterns from audio recordings. It's designed for leak detection and water flow monitoring applications.

## Features

- **Audio Feature Extraction**: Extracts 52 acoustic features from WAV files
- **Multiple ML Models**: Trains both Random Forest and SVM classifiers
- **Model Persistence**: Saves trained models for deployment
- **Comprehensive Evaluation**: Provides detailed performance metrics and visualizations
- **Standalone Predictor**: Creates a ready-to-use prediction script
- **Production Ready**: Complete workflow from training to deployment

## Installation Requirements

```bash
pip install numpy librosa matplotlib scikit-learn seaborn joblib
Project Structure
text
water-flow-classifier/
├── train_and_save_models.py  # Main training script
├── standalone_predictor.py   # Generated prediction script
├── saved_models/             # Directory for trained models
│   ├── ml_classifier.joblib
│   ├── feature_extractor_config.joblib
│   └── model_info.joblib
└── train/                    # Training data directory
    ├── idle_*.wav
    ├── normal_*.wav
    └── leakage_*.wav
Audio Features Extracted
The system extracts 52 acoustic features including:

MFCCs (13 mean + 13 standard deviation)

Spectral Features: Centroid, rolloff (mean + std)

Zero Crossing Rate (mean + std)

RMS Energy (mean + std)

Spectral Contrast (7 bands)

Chroma Features (12 notes)

Model Training
Data Preparation
Audio files should follow this naming convention:

idle_*.wav - No water flow

normal_*.wav - Regular water flow

leakage_*.wav - Water leakage sounds

Training Process
Feature Extraction: Convert audio files to feature vectors

Data Splitting: 80% training, 20% testing (stratified)

Model Training:

Random Forest (100 estimators)

SVM (RBF kernel)

Evaluation: Classification reports and confusion matrices

Model Saving: Export all components as joblib files

Running Training
python
from train_and_save_models import complete_workflow

# Run complete training pipeline
complete_workflow()
Or run specific components:

python
from train_and_save_models import train_models, save_trained_models

# Train models only
ml_classifier, feature_extractor = train_models("/path/to/audio/data")

# Save trained models
save_trained_models(ml_classifier, feature_extractor)
Prediction
Using the Standalone Predictor
bash
python standalone_predictor.py "path/to/audio/file.wav"
Optional: Specify model type

bash
python standalone_predictor.py "audio.wav" random_forest
python standalone_predictor.py "audio.wav" svm
Programmatic Prediction
python
from train_and_save_models import load_models, predict_new_audio

# Load saved models
ml_classifier, feature_extractor = load_models("saved_models")

# Predict on new audio
prediction, probabilities = predict_new_audio(
    "path/to/audio.wav", 
    ml_classifier, 
    feature_extractor
)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
Output Example
text
PREDICTION RESULTS
File: test_audio.wav
Prediction: LEAKAGE
Probabilities:
   idle: 0.0234
   normal_flow: 0.1567
   leakage: 0.8199
Confidence: 81.99% (HIGH)
Model Performance
The system provides comprehensive evaluation:

Classification Reports: Precision, recall, F1-score

Confusion Matrices: Visual representation of predictions

Feature Importance: Top 10 most important acoustic features

Probability Scores: Confidence levels for each prediction

Configuration
Audio Processing Parameters
Sample rate: 22050 Hz

Duration: 2 seconds

Hop length: 512 samples

Model Parameters
Random Forest: 100 estimators, random state=42

SVM: RBF kernel, probability=True, random state=42

Error Handling
Automatic detection of missing data directories

Graceful handling of corrupted audio files

Comprehensive error messages for debugging

File validation and skip processing for problematic files

Use Cases
Water Leak Detection: Identify potential leaks in plumbing systems

Flow Monitoring: Classify normal vs abnormal water flow patterns

Conservation Analysis: Monitor water usage through audio patterns

Smart Home Integration: Integrate with home automation systems

File Naming Convention
For automatic label detection, name files with these keywords:

idle - No water flow

normal - Regular water flow

leakage - Water leakage

Example: water_flow_leakage_001.wav, idle_kitchen_faucet.wav

Troubleshooting
Common Issues
No audio files found: Check directory path and file extensions (.wav)

Feature extraction errors: Verify audio file integrity and format

Model loading failures: Ensure all joblib files are in saved_models directory

Low accuracy: Check audio quality and ensure proper labeling

Support
For issues with the classification system, check:

Audio file quality and length

Consistent naming conventions

Sufficient training data per class

Proper installation of dependencies

License
This water flow audio classification system is designed for educational and research purposes in acoustic pattern recognition and machine learning applications.



Summary of Real-time Water Flow Leakage Detection System
Overview
This Python-based system provides continuous real-time analysis of water flow sounds to detect potential leaks using machine learning and audio processing. The system processes audio from a specified WAV file and generates multiple visualizations until manually stopped by the user.

Core Components
1. Continuous Audio Analyzer (ContinuousAudioAnalyzer)
Purpose: Manages continuous audio playback and processing

Key Features:

Loops the WAV file indefinitely for uninterrupted analysis

Generates randomized audio chunks by adding noise and variations to original audio

Thread-safe operations for real-time data access

Runs in background until user interruption (Ctrl+C)

2. Water Flow Visualization (WaterFlowGenerator)
Purpose: Creates real-time water flow animation based on audio characteristics

Key Features:

Generates flowing water profile that responds to audio amplitude

Smooth, physics-like water movement with natural variations

Real-time updates synchronized with audio playback

Visual representation of water levels and flow patterns

3. Leakage Detection System (RealtimeLeakageVisualizer)
Purpose: Machine learning-based detection of water leakage patterns

Key Features:

Loads pre-trained ML models (Random Forest, SVM) for classification

Extracts comprehensive audio features (MFCC, spectral, ZCR, RMS, chroma)

Provides real-time predictions with confidence scores

Falls back to simulation mode if models are unavailable

Analysis Modes
Mode 1: Water Flow Visualization Only
Continuous animated water flow representation

Real-time response to audio amplitude changes

Simple, intuitive visualization of flow patterns

Mode 2: Leakage Detection Dashboard
Comprehensive real-time monitoring dashboard with 6 subplots:

Current Audio Waveform - Live audio visualization

Current Probabilities - Pie chart of class probabilities

Probability Timeline - Historical probability trends

Leakage Indicator - Visual leakage probability meter

Confidence Meter - Prediction confidence level

Status Panel - Current detection status

Mode 3: Combined Analysis
Sequential or parallel execution of both visualization systems

Comprehensive monitoring of both flow patterns and leakage detection

Technical Features
Audio Processing
Sample Rate: 22.05 kHz standard processing

Feature Extraction: 53-dimensional feature vector including:

MFCC coefficients (mean + std)

Spectral centroid and rolloff

Zero Crossing Rate (ZCR)

RMS energy

Spectral contrast

Chroma features

Machine Learning
Classification: Three-class system (idle, leakage, normal_flow)

Models: Random Forest and SVM with ensemble capability

Preprocessing: Standard scaling and label encoding

Threshold: 60% probability for leakage alerts

Real-time Capabilities
Continuous Operation: Runs indefinitely until user interruption

Live Updates: All visualizations update in real-time

Performance Optimized: Limited history buffers for smooth operation

Thread Management: Separate threads for audio playback and analysis

User Interface
Console Output
Real-time status updates with timestamps

Color-coded alerts (🔴 Leakage, 🟢 Normal, 🟡 Idle)

Probability distributions for each class

Sample counting and confidence metrics

Visual Dashboard
Professional matplotlib/seaborn styling

Multiple coordinated subplots

Color-coded indicators and thresholds

Real-time waveform display

File Specifications
Input: C:\a_python\accoustics\leakagen\leakagesoundgenerator.wav

Models: saved_models/ directory containing:

ml_classifier.joblib - Trained ML models

feature_extractor_config.joblib - Feature extraction configuration

Key Benefits
Continuous Monitoring: Operates 24/7 without interruption

Real-time Detection: Immediate leakage alerts

Multiple Visualization Options: Flexible analysis modes

Robust Error Handling: Graceful degradation without models

Professional Output: Comprehensive reports and visualizations

User-friendly: Simple mode selection and clear status indicators

Use Cases
Water utility monitoring systems

Building maintenance and leak detection

Industrial pipeline monitoring

Research and development in acoustic leak detection

Educational demonstrations of audio ML applications

The system provides a complete solution for acoustic-based water leakage detection, combining advanced machine learning with intuitive real-time visualizations for continuous monitoring applications.
