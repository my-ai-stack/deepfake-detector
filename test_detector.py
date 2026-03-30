"""Tests for Deepfake Detector"""
import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_detector_import():
    """Test detector module imports"""
    from detector import DeepfakeDetector, AudioFeatureExtractor
    assert DeepfakeDetector is not None


def test_detector_init():
    """Test DeepfakeDetector initialization"""
    from detector import DeepfakeDetector
    detector = DeepfakeDetector()
    assert detector is not None


def test_feature_extractor():
    """Test AudioFeatureExtractor initialization"""
    from detector import AudioFeatureExtractor
    extractor = AudioFeatureExtractor()
    assert extractor is not None
    assert extractor.sample_rate == 22050


def test_detector_analyze_nonexistent():
    """Test analyze returns error for non-existent file"""
    from detector import DeepfakeDetector
    detector = DeepfakeDetector()
    result = detector.analyze_audio("nonexistent.wav")
    assert result["status"] == "error"


def test_analyze_features():
    """Test feature analysis"""
    from detector import DeepfakeDetector
    detector = DeepfakeDetector()
    features = {
        "spectral_centroid": 1000,
        "mfcc_std": 10,
        "zero_crossing_rate": 0.1,
        "rms_energy": 0.2
    }
    analysis = detector._analyze_features(features)
    assert "is_deepfake" in analysis
    assert "confidence" in analysis