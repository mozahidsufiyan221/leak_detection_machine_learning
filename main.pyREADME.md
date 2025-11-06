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
