"""
Test RVC Logic

Verifies the split-awareness of the RVC selection logic in SynthesizerManager.
This is a mock test that does not require RVC models to be downloaded or rvc-python to be installed.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

class TestRVCLogin(unittest.TestCase):
    
    def setUp(self):
        # Mock Config
        from pipeline.config import Config, SynthesisConfig, VCPoolsConfig
        
        self.mock_config = MagicMock(spec=Config)
        self.mock_config.synthesis = MagicMock(spec=SynthesisConfig)
        self.mock_config.synthesis.enable_vc = True
        self.mock_config.synthesis.vc_device = "cpu"
        
        # Setup Pools
        self.mock_config.synthesis.vc_pools = VCPoolsConfig(
            train=["train_model_1.pth", "train_model_2.pth"],
            test=["test_model_1.pth"],
            val=[] # Empty val pool
        )
        
        # Mock SynthesizerManager with minimal initialization
        from pipeline.synthesizer.synthesizer import SynthesizerManager
        self.manager = SynthesizerManager(self.mock_config)
        self.manager.vc_synthesizers = [] # No real synthesizers loaded
        
    @patch("pipeline.synthesizer.synthesizer.random.choice")
    @patch("pipeline.utils.pth_security.is_safe_to_load")
    @patch("pipeline.synthesizer.rvc_synthesizer.synthesize_vc")
    def test_split_selection_train(self, mock_synthesize, mock_is_safe, mock_choice):
        """Test that TRAIN split selects from train pool."""
        # Setup mocks
        mock_choice.side_effect = lambda x: x[0] # Always pick first item
        mock_is_safe.return_value = True
        mock_synthesize.return_value = [0.1, 0.2] # Dummy audio
        
        # Run logic
        result = self.manager._synthesize_vc_split_aware(
            audio=None, sr=16000, split="train"
        )
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['method'], 'vc')
        # Should pick from train pool
        self.assertIn("train_model_1.pth", result['generator']) 
        
    @patch("pipeline.synthesizer.synthesizer.random.choice")
    @patch("pipeline.utils.pth_security.is_safe_to_load")
    @patch("pipeline.synthesizer.rvc_synthesizer.synthesize_vc")
    def test_split_selection_test(self, mock_synthesize, mock_is_safe, mock_choice):
        """Test that TEST split selects from test pool."""
        mock_choice.side_effect = lambda x: x[0]
        mock_is_safe.return_value = True
        mock_synthesize.return_value = [0.1, 0.2]
        
        result = self.manager._synthesize_vc_split_aware(
            audio=None, sr=16000, split="test"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("test_model_1.pth", result['generator'])

    def test_unknown_split_rejection(self):
        """Test that unknown split returns None (skips VC)."""
        result = self.manager._synthesize_vc_split_aware(
            audio=None, sr=16000, split="unknown_split"
        )
        self.assertIsNone(result)

    def test_none_split_rejection(self):
        """Test that None split returns None."""
        result = self.manager._synthesize_vc_split_aware(
            audio=None, sr=16000, split=None
        )
        self.assertIsNone(result)

    def test_empty_pool_handling(self):
        """Test that empty pool fails gracefully."""
        result = self.manager._synthesize_vc_split_aware(
            audio=None, sr=16000, split="val" # Val pool is empty
        )
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
