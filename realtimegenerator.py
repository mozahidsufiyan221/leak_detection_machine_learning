import os
import numpy as np
import librosa
import librosa.display
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pygame
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from collections import deque
import threading
import random

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ContinuousAudioAnalyzer:
    def __init__(self, wav_file_path):
        self.wav_file_path = wav_file_path
        self.is_running = False
        self.current_audio_chunk = None
        self.analysis_results = []
        self.lock = threading.Lock()
        
    def start_continuous_analysis(self):
        """Start continuous audio analysis"""
        self.is_running = True
        
        # Start audio playback in a loop
        def audio_playback():
            pygame.mixer.init()
            while self.is_running:
                try:
                    pygame.mixer.music.load(self.wav_file_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and self.is_running:
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Audio playback error: {e}")
                    time.sleep(1)
        
        # Start audio playback thread
        audio_thread = threading.Thread(target=audio_playback)
        audio_thread.daemon = True
        audio_thread.start()
        
        print("🔄 Continuous analysis started. Press Ctrl+C to stop...")
        
        try:
            while self.is_running:
                # Simulate random audio chunk generation
                self._generate_random_audio_chunk()
                time.sleep(0.1)  # Small delay between analyses
                
        except KeyboardInterrupt:
            self.stop_analysis()
    
    def _generate_random_audio_chunk(self):
        """Generate random audio chunk for analysis"""
        # Load original audio for reference
        try:
            sample_rate, original_audio = wavfile.read(self.wav_file_path)
            if len(original_audio.shape) > 1:
                original_audio = np.mean(original_audio, axis=1)
            
            # Create random chunk by taking random segment and adding noise
            chunk_size = min(22050, len(original_audio))  # 1 second at 22.05kHz
            
            if len(original_audio) > chunk_size:
                start_idx = random.randint(0, len(original_audio) - chunk_size)
                base_chunk = original_audio[start_idx:start_idx + chunk_size]
            else:
                base_chunk = original_audio
            
            # Add random noise and variations
            noise_level = random.uniform(0.01, 0.1)
            random_noise = np.random.normal(0, noise_level, len(base_chunk))
            
            # Random amplitude variation
            amplitude_variation = random.uniform(0.5, 1.5)
            
            # Create final chunk with variations
            random_chunk = (base_chunk * amplitude_variation + random_noise)
            random_chunk = np.clip(random_chunk, -1.0, 1.0)
            
            with self.lock:
                self.current_audio_chunk = random_chunk
                
        except Exception as e:
            print(f"Error generating audio chunk: {e}")
    
    def get_current_chunk(self):
        """Get the current audio chunk for analysis"""
        with self.lock:
            return self.current_audio_chunk.copy() if self.current_audio_chunk is not None else None
    
    def stop_analysis(self):
        """Stop continuous analysis"""
        self.is_running = False
        pygame.mixer.music.stop()
        print("\n🛑 Analysis stopped by user")

class WaterFlowGenerator:
    def __init__(self, audio_analyzer):
        """
        Initialize the water flow generator with continuous audio analyzer.
        """
        self.audio_analyzer = audio_analyzer
        self.is_running = False
        
        # Initialize parameters
        self.water_levels = deque([0.5] * 100, maxlen=100)
        self.smoothing_factor = 0.7
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.8)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, 100)
        self.ax.set_title('Real-time Water Flow Profile - Continuous Analysis')
        self.ax.set_ylabel('Water Level')
        self.ax.set_xlabel('Time Steps')
        self.ax.grid(True, alpha=0.3)
        
    def get_audio_amplitude(self):
        """Get current audio amplitude from analyzer"""
        chunk = self.audio_analyzer.get_current_chunk()
        if chunk is not None:
            rms = np.sqrt(np.mean(chunk**2))
            return rms
        return 0.5  # Default value
    
    def update_water_levels(self, amplitude):
        """Update water levels based on audio amplitude."""
        # Apply smoothing and add some randomness for natural water movement
        new_level = (self.smoothing_factor * self.water_levels[-1] + 
                    (1 - self.smoothing_factor) * amplitude + 
                    np.random.normal(0, 0.02))
        
        # Ensure water level stays within bounds
        new_level = max(0.1, min(0.9, new_level))
        
        self.water_levels.append(new_level)
    
    def update_plot(self, frame):
        """Update the plot for animation."""
        if not self.audio_analyzer.is_running:
            return self.line,
            
        # Get audio amplitude
        amplitude = self.get_audio_amplitude()
        
        # Update water levels
        self.update_water_levels(amplitude)
        
        # Update the plot data
        x_data = list(range(len(self.water_levels)))
        y_data = list(self.water_levels)
        
        self.line.set_data(x_data, y_data)
        
        # Update fill area
        self.ax.collections.clear()
        self.ax.fill_between(x_data, 0, y_data, color='cyan', alpha=0.3)
        
        # Update title with current amplitude
        self.ax.set_title(f'Real-time Water Flow Profile - Amplitude: {amplitude:.3f}')
        
        return self.line,
    
    def start_animation(self):
        """Start the animation."""
        self.is_running = True
        ani = FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False
        )
        plt.show()
        
    def close(self):
        """Clean up resources."""
        self.is_running = False

class RealtimeLeakageVisualizer:
    def __init__(self, audio_analyzer, models_dir="saved_models"):
        self.audio_analyzer = audio_analyzer
        self.models_dir = models_dir
        self.ml_classifier = None
        self.feature_extractor = None
        self.results = []
        self.is_running = False
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            if not os.path.exists(self.models_dir):
                print("❌ Models directory not found. Running in simulation mode.")
                return False
                
            model_data = joblib.load(os.path.join(self.models_dir, 'ml_classifier.joblib'))
            feature_config = joblib.load(os.path.join(self.models_dir, 'feature_extractor_config.joblib'))
            
            class LoadedClassifier:
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
                    return self.label_encoder.inverse_transform(predictions_encoded)
                
                def predict_proba(self, X, model_name='random_forest'):
                    X_scaled = self.scaler.transform(X)
                    model = self.models[model_name]
                    return model.predict_proba(X_scaled)
            
            class FeatureExtractor:
                def __init__(self, config):
                    self.sample_rate = config['sample_rate']
                    self.duration = config['duration']
                    self.hop_length = config['hop_length']
                
                def extract_features(self, audio_data, sr):
                    try:
                        features = []
                        
                        # MFCC
                        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=self.hop_length)
                        features.extend(np.mean(mfcc, axis=1))
                        features.extend(np.std(mfcc, axis=1))
                        
                        # Spectral features
                        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=self.hop_length)
                        features.append(np.mean(spectral_centroid))
                        features.append(np.std(spectral_centroid))
                        
                        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=self.hop_length)
                        features.append(np.mean(spectral_rolloff))
                        features.append(np.std(spectral_rolloff))
                        
                        # ZCR
                        zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)
                        features.append(np.mean(zcr))
                        features.append(np.std(zcr))
                        
                        # RMS
                        rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)
                        features.append(np.mean(rms))
                        features.append(np.std(rms))
                        
                        # Spectral contrast
                        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, hop_length=self.hop_length)
                        features.extend(np.mean(contrast, axis=1))
                        
                        # Chroma
                        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=self.hop_length)
                        features.extend(np.mean(chroma, axis=1))
                        
                        return np.array(features)
                    except Exception as e:
                        print(f"Feature extraction error: {e}")
                        return None
            
            self.ml_classifier = LoadedClassifier(model_data)
            self.feature_extractor = FeatureExtractor(feature_config)
            print("✅ Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("Running in simulation mode with random predictions.")
            return False
    
    def predict_segment(self, audio_data, sr=22050):
        """Predict class for audio segment"""
        if self.ml_classifier is None:
            # Simulation mode - generate random predictions
            classes = ['idle', 'leakage', 'normal_flow']
            prediction = random.choice(classes)
            probabilities = np.random.dirichlet(np.ones(3), size=1)[0]
            return prediction, probabilities
        
        features = self.feature_extractor.extract_features(audio_data, sr)
        if features is not None and len(features) == 53:
            features = features.reshape(1, -1)
            prediction = self.ml_classifier.predict(features)[0]
            probabilities = self.ml_classifier.predict_proba(features)[0]
            return prediction, probabilities
        return None, None
    
    def create_realtime_dashboard(self):
        """Create real-time continuous analysis dashboard"""
        # Setup real-time plot
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Continuous Real-Time Water Flow Leakage Detection', fontsize=16, fontweight='bold')
        
        # Create subplots
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # Current audio waveform
        ax2 = plt.subplot2grid((3, 3), (0, 2))             # Current probabilities
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)  # Probability timeline
        ax4 = plt.subplot2grid((3, 3), (2, 0))             # Leakage indicator
        ax5 = plt.subplot2grid((3, 3), (2, 1))             # Confidence
        ax6 = plt.subplot2grid((3, 3), (2, 2))             # Status
        
        plt.tight_layout(pad=3.0)
        
        # Initialize data storage
        self.results = []
        times = []
        prob_history = {'idle': [], 'leakage': [], 'normal_flow': []}
        start_time = time.time()
        
        self.is_running = True
        
        def update_dashboard(frame):
            if not self.audio_analyzer.is_running:
                return []
                
            # Get current audio chunk
            audio_chunk = self.audio_analyzer.get_current_chunk()
            if audio_chunk is None or len(audio_chunk) < 512:
                return []
            
            current_time = time.time() - start_time
            
            # Get prediction
            prediction, probabilities = self.predict_segment(audio_chunk)
            
            if prediction is not None:
                # Store results
                result = {
                    'time': current_time,
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'timestamp': time.strftime("%H:%M:%S")
                }
                self.results.append(result)
                
                # Update probability history
                classes = ['idle', 'leakage', 'normal_flow']
                for j, cls in enumerate(classes):
                    if j < len(probabilities):
                        prob_history[cls].append(probabilities[j])
                times.append(current_time)
                
                # Keep only last 100 points for performance
                if len(times) > 100:
                    times.pop(0)
                    for cls in classes:
                        if len(prob_history[cls]) > 100:
                            prob_history[cls].pop(0)
                
                # Update plots
                self._update_plots(ax1, ax2, ax3, ax4, ax5, ax6, audio_chunk, times, prob_history, result)
                
                # Console output
                self._print_segment_result(result, len(self.results))
            
            return []
        
        # Start animation
        ani = FuncAnimation(fig, update_dashboard, interval=1000, blit=False, cache_frame_data=False)
        plt.show()
        
        self.is_running = False
    
    def _update_plots(self, ax1, ax2, ax3, ax4, ax5, ax6, audio_chunk, times, prob_history, current_result):
        """Update all plots with current data"""
        # Clear all axes
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.clear()
        
        # Plot 1: Current audio waveform
        ax1.plot(audio_chunk, 'b-', alpha=0.7, linewidth=0.5)
        ax1.set_xlim(0, len(audio_chunk))
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Current Audio Waveform', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Current probabilities (pie chart)
        classes = ['idle', 'leakage', 'normal_flow']
        colors = ['#FFD700', '#FF6B6B', '#4ECDC4']
        current_probs = current_result['probabilities']
        
        wedges, texts, autotexts = ax2.pie(current_probs, labels=classes, 
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title('Current Probabilities', fontweight='bold')
        
        # Plot 3: Probability timeline
        if len(times) > 1:
            for cls, color in zip(classes, colors):
                if len(prob_history[cls]) == len(times):
                    ax3.plot(times, prob_history[cls], label=cls, color=color, linewidth=2)
            
            ax3.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Leakage Threshold')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Probability')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.set_title('Probability Timeline (Last 100 samples)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Leakage indicator
        leakage_prob = current_result['probabilities'][1]  # leakage probability
        color = 'red' if leakage_prob > 0.6 else 'orange' if leakage_prob > 0.3 else 'green'
        ax4.bar(['Leakage'], [leakage_prob], color=color, alpha=0.8, edgecolor='black')
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('Probability')
        ax4.set_title('Leakage Indicator', fontweight='bold')
        ax4.text(0, leakage_prob/2, f'{leakage_prob:.3f}', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Confidence meter
        confidence = max(current_result['probabilities'])
        ax5.bar(['Confidence'], [confidence], color='blue', alpha=0.8, edgecolor='black')
        ax5.set_ylim(0, 1)
        ax5.set_ylabel('Confidence')
        ax5.set_title('Prediction Confidence', fontweight='bold')
        ax5.text(0, confidence/2, f'{confidence:.3f}', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Status
        prediction = current_result['prediction']
        leakage_prob = current_result['probabilities'][1]
        status_color = 'red' if prediction == 'leakage' and leakage_prob > 0.6 else 'green'
        ax6.text(0.5, 0.6, 'CURRENT\nSTATUS', ha='center', va='center', 
                fontweight='bold', fontsize=14, transform=ax6.transAxes)
        ax6.text(0.5, 0.3, f'{prediction.upper()}', ha='center', va='center', 
                fontweight='bold', fontsize=16, color=status_color, transform=ax6.transAxes)
        ax6.text(0.5, 0.1, f'Samples: {len(self.results)}', ha='center', va='center',
                fontweight='bold', fontsize=10, transform=ax6.transAxes)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Current Status', fontweight='bold')
    
    def _print_segment_result(self, result, sample_count):
        """Print segment result to console"""
        classes = ['idle', 'leakage', 'normal_flow']
        current_time = result['timestamp']
        
        # Create formatted output
        prob_str = " | ".join([f"{cls}: {prob:.3f}" for cls, prob in zip(classes, result['probabilities'])])
        
        if result['prediction'] == 'leakage' and result['probabilities'][1] > 0.6:
            status = "🔴 LEAKAGE"
        elif result['prediction'] == 'normal_flow':
            status = "🟢 NORMAL"
        else:
            status = "🟡 IDLE"
        
        print(f"{current_time} [Sample:{sample_count:04d}] {status:12} | {prob_str}")
    
    def close(self):
        """Clean up resources."""
        self.is_running = False

def main():
    """Main execution function"""
    print("🌊 CONTINUOUS REAL-TIME WATER FLOW LEAKAGE DETECTOR")
    print("===================================================")
    
    # Define the WAV file path
    wav_file_path = r"C:\a_python\accoustics\leakagen\leakagesoundgenerator.wav"
    
    if not os.path.exists(wav_file_path):
        print(f"❌ WAV file not found: {wav_file_path}")
        print("Please check the file path and try again.")
        return
    
    try:
        # Initialize continuous audio analyzer
        audio_analyzer = ContinuousAudioAnalyzer(wav_file_path)
        
        # Check if models exist
        models_exist = os.path.exists("saved_models")
        
        print("\nSelect continuous analysis mode:")
        print("1. Water Flow Visualization Only")
        if models_exist:
            print("2. Leakage Detection Dashboard")
            print("3. Both (Water Flow + Leakage Detection)")
        else:
            print("2. Leakage Detection Dashboard (Simulation Mode)")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        # Start continuous audio analysis in background
        analysis_thread = threading.Thread(target=audio_analyzer.start_continuous_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # Give analyzer time to start
        time.sleep(2)
        
        if choice == "1":
            # Water flow visualization only
            print("\n🚰 Starting Continuous Water Flow Visualization...")
            generator = WaterFlowGenerator(audio_analyzer)
            generator.start_animation()
            generator.close()
            
        elif choice == "2":
            # Leakage detection only
            print("\n🔍 Starting Continuous Leakage Detection Analysis...")
            visualizer = RealtimeLeakageVisualizer(audio_analyzer)
            visualizer.create_realtime_dashboard()
            visualizer.close()
            
        elif choice == "3" and models_exist:
            # Both visualizations (simplified - run sequentially)
            print("\n🚰🔍 Starting Combined Analysis...")
            print("Note: Running water flow visualization first...")
            
            generator = WaterFlowGenerator(audio_analyzer)
            
            def run_both():
                generator.start_animation()
            
            # Start in thread
            both_thread = threading.Thread(target=run_both)
            both_thread.daemon = True
            both_thread.start()
            
            # Wait for user to close water flow, then start leakage detection
            try:
                while both_thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
                
            print("\n🔄 Switching to Leakage Detection...")
            visualizer = RealtimeLeakageVisualizer(audio_analyzer)
            visualizer.create_realtime_dashboard()
            visualizer.close()
            generator.close()
            
        else:
            print("❌ Invalid choice.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Program interrupted by user")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        audio_analyzer.stop_analysis()
        print("🎯 Analysis completed.")

if __name__ == "__main__":
    main()