# Real-time Water Flow Leakage Detection System

## 📋 Overview

A Python-based system for **continuous real-time analysis** of water flow sounds to detect potential leaks using machine learning and audio processing. The system processes audio from a specified WAV file and generates multiple visualizations until manually stopped by the user.

![System Architecture](https://img.shields.io/badge/Architecture-Modular-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-green)
![ML](https://img.shields.io/badge/ML-Enabled-orange)

## 🏗️ System Architecture

### Core Components

#### 1. Continuous Audio Analyzer (`ContinuousAudioAnalyzer`)
**Purpose**: Manages continuous audio playback and processing

**Key Features**:
- 🔁 **Looping Playback**: Continuously plays WAV file indefinitely
- 🎵 **Random Chunk Generation**: Creates randomized audio segments with noise/variations
- 🔒 **Thread-Safe Operations**: Uses locks for safe data access between threads
- ⏹️ **User Interruption**: Runs until Ctrl+C command

#### 2. Water Flow Visualization (`WaterFlowGenerator`)
**Purpose**: Creates real-time water flow animation based on audio characteristics

**Key Features**:
- 💧 **Physics-like Movement**: Natural water flow simulation
- 📊 **Amplitude Response**: Visual changes based on audio amplitude
- 🔄 **Real-time Updates**: Continuous animation synchronized with audio
- 🎨 **Visual Representation**: Intuitive flow pattern display

#### 3. Leakage Detection System (`RealtimeLeakageVisualizer`)
**Purpose**: Machine learning-based detection of water leakage patterns

**Key Features**:
- 🤖 **Pre-trained Models**: Random Forest and SVM classifiers
- 🔊 **Feature Extraction**: Comprehensive audio analysis (53 features)
- ⚡ **Real-time Predictions**: Live classification with confidence scores
- 🎮 **Simulation Mode**: Fallback operation without trained models

## 🎯 Analysis Modes

### Mode 1: Water Flow Visualization Only
- **Continuous animated water flow** representation
- **Real-time response** to audio amplitude changes
- **Simple, intuitive** visualization of flow patterns

### Mode 2: Leakage Detection Dashboard
**6-Panel Real-time Monitoring Dashboard**:

| Panel | Description | Visualization |
|-------|-------------|---------------|
| 1. **Current Audio Waveform** | Live audio signal display | Line plot |
| 2. **Current Probabilities** | Class probability distribution | Pie chart |
| 3. **Probability Timeline** | Historical probability trends | Multi-line plot |
| 4. **Leakage Indicator** | Visual leakage probability meter | Bar chart |
| 5. **Confidence Meter** | Prediction confidence level | Bar chart |
| 6. **Status Panel** | Current detection status | Text display |

### Mode 3: Combined Analysis
- **Sequential execution** of both systems
- **Comprehensive monitoring** of flow patterns and leakage detection
- **Flexible operation** modes

## 🔧 Technical Specifications

### Audio Processing Pipeline
Audio File → Continuous Playback → Random Chunk Generation
↓
Feature Extraction → ML Classification → Real-time Visualization
↓
Dashboard Updates → Console Logging → User Interaction

text

### Feature Extraction (53 Dimensions)
- **MFCC Features**: 13 mean + 13 standard deviation coefficients
- **Spectral Features**: Centroid & rolloff (mean + std)
- **Zero Crossing Rate**: Mean and standard deviation
- **RMS Energy**: Mean and standard deviation
- **Spectral Contrast**: 7 frequency bands
- **Chroma Features**: 12 pitch classes

### Machine Learning Configuration
- **Classification Types**: `idle`, `leakage`, `normal_flow`
- **Algorithms**: Random Forest + SVM ensemble
- **Threshold**: 60% probability for leakage alerts
- **Preprocessing**: StandardScaler + LabelEncoder
- **Confidence Tracking**: Continuous monitoring

### Performance Characteristics
| Parameter | Specification |
|-----------|---------------|
| Update Interval | 50-100ms |
| Audio Buffer | 1024 samples |
| Sample Rate | 22.05 kHz |
| History Window | 100 samples |
| Feature Vector | 53 dimensions |

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Audio playback capability
- Sufficient RAM for real-time processing

### Dependencies Installation
```bash
pip install numpy matplotlib pygame scipy librosa joblib seaborn scikit-learn
Required File Structure
text
project/
├── continuous_leakage_detector.py
├── saved_models/
│   ├── ml_classifier.joblib
│   └── feature_extractor_config.joblib
└── audio_files/
    └── leakagesoundgenerator.wav
🎮 Usage Guide
Starting the System
bash
python continuous_leakage_detector.py
Operation Flow
System Initialization

Load ML models (if available)

Initialize audio processing

Setup visualization frameworks

Mode Selection

text
Select continuous analysis mode:
1. Water Flow Visualization Only
2. Leakage Detection Dashboard
3. Both (Water Flow + Leakage Detection)

Enter your choice (1-3):
Continuous Operation

Real-time audio analysis begins

Visualizations update continuously

Console shows live status updates

System Shutdown

Press Ctrl+C to stop analysis

Graceful cleanup of resources

Generation of final reports

Expected Console Output
plaintext
🌊 CONTINUOUS REAL-TIME WATER FLOW LEAKAGE DETECTOR
===================================================

🔄 Continuous analysis started. Press Ctrl+C to stop...
🕒 14:30:25 [Sample:0042] 🔴 LEAKAGE     | idle: 0.150 | leakage: 0.720 | normal_flow: 0.130
🕒 14:30:26 [Sample:0043] 🟢 NORMAL      | idle: 0.100 | leakage: 0.250 | normal_flow: 0.650
🕒 14:30:27 [Sample:0044] 🟡 IDLE        | idle: 0.450 | leakage: 0.300 | normal_flow: 0.250
📊 Output & Reporting
Real-time Console Output
Timestamped entries with sample numbers

Color-coded status indicators:

🔴 LEAKAGE (probability > 60%)

🟢 NORMAL (normal_flow classification)

🟡 IDLE (idle classification)

Probability distributions for all classes

Confidence metrics and prediction details

Visual Dashboard Features
Professional styling with matplotlib/seaborn

Coordinated multi-plot updates

Color-coded thresholds and indicators

Real-time waveform display

Historical trend analysis

Generated Reports
PNG summary files: leakage_summary_[filename].png

Statistical analysis of detection results

Probability timelines and confidence distributions

Final assessment with leakage ratio calculations

🛠️ Technical Implementation
Class Structure
ContinuousAudioAnalyzer
python
class ContinuousAudioAnalyzer:
    def start_continuous_analysis(self)
    def _generate_random_audio_chunk(self)
    def get_current_chunk(self)
    def stop_analysis(self)
WaterFlowGenerator
python
class WaterFlowGenerator:
    def get_audio_amplitude(self)
    def update_water_levels(self, amplitude)
    def update_plot(self, frame)
    def start_animation(self)
RealtimeLeakageVisualizer
python
class RealtimeLeakageVisualizer:
    def load_models(self)
    def predict_segment(self, audio_data, sr=22050)
    def create_realtime_dashboard(self)
    def _update_plots(self, ...)
Thread Management
Audio playback in separate thread

Real-time visualization updates

Thread-safe data access with locking mechanisms

Graceful shutdown handling

Error Handling
Model loading fallbacks to simulation mode

Audio file validation and error reporting

Resource cleanup on interruption

Comprehensive exception handling

💡 Use Cases & Applications
🏢 Water Utility Monitoring
Continuous pipeline monitoring in distribution networks

Early leak detection for preventive maintenance

Resource conservation through timely interventions

🏢 Building Management
Plumbing system monitoring in commercial/residential buildings

Leak prevention in high-value facilities

Maintenance scheduling based on acoustic analysis

🏭 Industrial Applications
Pipeline integrity monitoring in manufacturing

Process control and quality assurance

Equipment monitoring for predictive maintenance

🎓 Educational & Research
Acoustic analysis demonstrations for signal processing

ML pattern recognition case studies

Real-time system design examples

⚡ Performance Optimization
Memory Management
Circular buffers for history data (100 sample limit)

Efficient feature extraction with librosa

Optimized matplotlib updates for real-time performance

Processing Efficiency
Chunk-based analysis for continuous operation

Background audio playback with pygame

Selective visualization updates

Scalability Considerations
Modular design for component replacement

Configurable parameters for different use cases

Extensible feature extraction pipeline

🔧 Configuration Options
Audio Parameters
python
# Configurable settings
segment_duration = 2.0      # Analysis segment length
buffer_size = 1024          # Audio buffer size
sample_rate = 22050         # Processing sample rate
smoothing_factor = 0.7      # Water flow smoothing
ML Thresholds
python
leakage_threshold = 0.6     # Leakage probability threshold
confidence_threshold = 0.8  # Minimum confidence level
history_window = 100        # Data points to maintain
🐛 Troubleshooting
Common Issues
Audio File Not Found

text
❌ WAV file not found: [file_path]
Please check the file path and try again.
Models Not Available

text
❌ Models directory not found. Running in simulation mode.
Performance Issues

Reduce history window size

Increase update intervals

Close other resource-intensive applications

Debug Mode
Enable verbose logging by modifying the console output functions to include additional debugging information.

📈 Future Enhancements
Planned Features
Web-based dashboard for remote monitoring

Database integration for historical analysis

Multiple audio source support

Advanced ML models (Neural Networks, etc.)

Mobile application for field use

Integration Possibilities
IoT sensor networks for distributed monitoring

API endpoints for system integration

Cloud processing for scalable analysis

Alert systems (email, SMS, etc.)

🤝 Contributing
Development Setup
Fork the repository

Create feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

Testing Guidelines
Test with various audio file formats

Verify real-time performance metrics

Validate ML model accuracy

Check memory usage patterns

📄 License
This project is licensed under the MIT License - see the LICENSE.md file for details.

🆘 Support
For support and questions:

Create an issue in the project repository

Check existing documentation and examples

Review troubleshooting section above

🎯 Key Benefits Summary
Benefit	Description	Impact
🚀 Continuous Operation	24/7 monitoring capability	Reduced manual oversight
⚡ Real-time Detection	Immediate leakage alerts	Faster response times
🎨 Multiple Visualizations	Flexible analysis options	Better situational awareness
🛡️ Robust Operation	Graceful error handling	Increased system reliability
📊 Professional Output	Comprehensive reporting	Informed decision making
👥 User-Friendly	Simple interface and controls	Reduced training requirements
📝 Conclusion
This system provides a complete acoustic-based water leakage detection solution that combines advanced machine learning with intuitive real-time visualizations. Ideal for continuous monitoring applications where early leak detection is critical for resource conservation and damage prevention.

The modular architecture allows for flexible deployment while maintaining robust performance characteristics suitable for both research and production environments.
