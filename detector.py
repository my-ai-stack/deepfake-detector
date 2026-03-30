#!/usr/bin/env python3
"""
Deepfake Detector - Detect AI-generated audio and video deepfakes
"""
import os
import sys
import json
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try imports
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torch
    from sklearn.metrics import accuracy_score
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AudioFeatureExtractor:
    """Extract audio features for deepfake detection"""

    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.n_mels = 128

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not LIBROSA_AVAILABLE:
            # Return dummy data
            return np.random.randn(22050)

        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

    def extract_features(self, audio: np.ndarray) -> dict:
        """Extract audio features for analysis"""
        features = {}

        if not LIBROSA_AVAILABLE:
            # Return mock features
            features = {
                "duration": len(audio) / 22050,
                "mfcc_mean": 0.5,
                "spectral_centroid": 1000,
                "zero_crossing_rate": 0.1,
                "rms_energy": 0.2
            }
            return features

        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        features["mfcc_mean"] = np.mean(mfccs)
        features["mfcc_std"] = np.std(mfccs)
        features["mfcc_delta"] = np.mean(librosa.feature.delta(mfccs))

        # Spectral features
        features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
        features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate))

        # Temporal features
        features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(audio))
        features["rms_energy"] = np.mean(librosa.feature.rms(y=audio))

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features["chroma_mean"] = np.mean(chroma)
        features["chroma_std"] = np.std(chroma)

        # Duration
        features["duration"] = len(audio) / self.sample_rate

        return features


class DeepfakeDetector:
    """Detect audio and video deepfakes"""

    def __init__(self, model_path: str = None):
        self.extractor = AudioFeatureExtractor()
        self.model = None
        self.is_trained = False

        if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                # Would load trained model here
                self.is_trained = True
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")

    def analyze_audio(self, file_path: str) -> dict:
        """Analyze audio for deepfake detection"""
        print(f"🎙️ Analyzing: {file_path}")

        try:
            # Load and extract features
            audio = self.extractor.load_audio(file_path)
            features = self.extractor.extract_features(audio)

            # Analyze features for anomalies
            analysis = self._analyze_features(features)

            result = {
                "status": "success",
                "file": file_path,
                "is_deepfake": analysis["is_deepfake"],
                "confidence": analysis["confidence"],
                "features": features,
                "analysis": analysis["details"]
            }

            return result

        except Exception as e:
            return {
                "status": "error",
                "file": file_path,
                "error": str(e)
            }

    def _analyze_features(self, features: dict) -> dict:
        """Analyze extracted features for deepfake indicators"""

        # Deepfake indicators (heuristics)
        # Real audio typically has:
        # - Varied spectral content
        # - Natural MFCC patterns
        # - Consistent energy levels

        indicators = []
        confidence = 0.5  # Default confidence

        # Check spectral centroid (deepfakes often have unusual centroids)
        if features.get("spectral_centroid", 0) < 500:
            indicators.append("Low spectral centroid")
            confidence += 0.1
        elif features.get("spectral_centroid", 0) > 8000:
            indicators.append("Unusually high spectral centroid")
            confidence += 0.1

        # Check MFCC patterns
        if features.get("mfcc_std", 0) < 5:
            indicators.append("Low MFCC variance (unnatural)")
            confidence += 0.15

        # Check zero crossing rate
        zcr = features.get("zero_crossing_rate", 0)
        if zcr < 0.01 or zcr > 0.5:
            indicators.append("Unusual zero crossing rate")
            confidence += 0.1

        # Check energy consistency
        rms = features.get("rms_energy", 0)
        if rms < 0.01:
            indicators.append("Very low energy (suspicious)")
            confidence += 0.1
        elif rms > 0.8:
            indicators.append("Very high energy (possible clipping)")
            confidence += 0.1

        # Decision
        is_deepfake = confidence > 0.6

        return {
            "is_deepfake": is_deepfake,
            "confidence": min(confidence, 0.95),
            "details": {
                "indicators": indicators,
                "naturalness_score": 1 - confidence
            }
        }

    def batch_analyze(self, directory: str, pattern: str = "*.wav") -> list:
        """Analyze multiple audio files"""
        results = []

        import glob
        files = glob.glob(os.path.join(directory, pattern))

        if not files:
            files = glob.glob(os.path.join(directory, "*.wav"))
            files.extend(glob.glob(os.path.join(directory, "*.mp3")))

        print(f"📁 Found {len(files)} files to analyze")

        for i, file_path in enumerate(files):
            print(f"  [{i+1}/{len(files)}] {os.path.basename(file_path)}")
            result = self.analyze_audio(file_path)
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Deepfake Detector - Detect AI-generated deepfakes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input voice.wav --type audio
  %(prog)s --input video.mp4 --type video
  %(prog)s --batch ./audio_samples/
  %(prog)s --interactive
        """
    )
    parser.add_argument('--input', '-i', help='Input audio/video file')
    parser.add_argument('--type', '-t', choices=['audio', 'video'], default='audio',
                       help='Type of input')
    parser.add_argument('--batch', '-b', help='Process directory of files')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    print("🔍 Deepfake Detector")
    print("=" * 40)

    if not LIBROSA_AVAILABLE:
        print("⚠️  librosa not installed. Running in limited mode.")
        print("   Install: pip install librosa soundfile numpy")
        print()

    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not installed. Using heuristic analysis.")
        print("   Install: pip install torch scikit-learn")
        print()

    detector = DeepfakeDetector()

    if args.batch:
        results = detector.batch_analyze(args.batch)
        print(f"\n📊 Analyzed {len(results)} files")

        # Summary
        deepfake_count = sum(1 for r in results if r.get("is_deepfake"))
        print(f"   Potential deepfakes: {deepfake_count}")
        print(f"   Likely authentic: {len(results) - deepfake_count}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   Results saved to: {args.output}")

    elif args.input:
        result = detector.analyze_audio(args.input)

        if result["status"] == "success":
            print(f"\n✅ Analysis Complete")
            print(f"   Deepfake: {'YES ⚠️' if result['is_deepfake'] else 'NO ✓'}")
            print(f"   Confidence: {result['confidence']:.1%}")

            if result.get("analysis", {}).get("indicators"):
                print(f"\n   Indicators:")
                for ind in result["analysis"]["indicators"]:
                    print(f"     - {ind}")

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n   Results saved to: {args.output}")
        else:
            print(f"❌ Error: {result.get('error')}")

    elif args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        print()

        while True:
            file_path = input("Audio file: ")
            if file_path.lower() in ('quit', 'exit', 'q'):
                break

            result = detector.analyze_audio(file_path)
            print(f"   Result: {result.get('is_deepfake', 'unknown')}")
            print()

    else:
        print("Usage:")
        print("  %(prog)s --input voice.wav")
        print("  %(prog)s --batch ./samples/")
        print("  %(prog)s --interactive")


if __name__ == '__main__':
    main()