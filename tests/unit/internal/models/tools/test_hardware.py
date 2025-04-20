"""
Tests for hardware detection tools.
"""

import unittest
from unittest.mock import patch, MagicMock

from plexe.internal.models.tools.hardware import get_gpu_info, get_ray_info


class TestGPUDetection(unittest.TestCase):
    """Tests for GPU detection utility functions."""
    
    @patch("importlib.util.find_spec")
    def test_pytorch_gpu_available(self, mock_find_spec):
        """Test GPU detection when PyTorch has GPU."""
        # Setup mock for importlib to allow import
        mock_find_spec.return_value = MagicMock()
        
        # Setup torch mock
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2
        
        # Patch torch import
        with patch.dict("sys.modules", {"torch": torch_mock}):
            result = get_gpu_info()
        
        self.assertTrue(result["available"])
        self.assertEqual(result["framework"], "pytorch")
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["count"], 2)
    
    @patch("importlib.util.find_spec")
    def test_gpu_not_available(self, mock_find_spec):
        """Test GPU detection when no GPUs are available."""
        # Mock all imports to return None (not found)
        mock_find_spec.return_value = None
        
        result = get_gpu_info()
        
        self.assertFalse(result["available"])
    
    @patch("importlib.util.find_spec")
    def test_tensorflow_gpu_available(self, mock_find_spec):
        """Test GPU detection when TensorFlow has GPU."""
        # Return None for PyTorch import
        def find_spec_side_effect(name):
            if name == "torch":
                return None
            return MagicMock()
        
        mock_find_spec.side_effect = find_spec_side_effect
        
        # Mock tensorflow
        tf_mock = MagicMock()
        tf_mock.config.list_physical_devices.return_value = [MagicMock(), MagicMock()]
        
        # Patch modules
        with patch.dict("sys.modules", {"tensorflow": tf_mock, "torch": None}):
            result = get_gpu_info()
        
        self.assertTrue(result["available"])
        self.assertEqual(result["framework"], "tensorflow")
        self.assertEqual(result["device"], "gpu")
        self.assertEqual(result["count"], 2)
    
    @patch("importlib.util.find_spec")
    def test_ray_info_available(self, mock_find_spec):
        """Test Ray cluster info detection."""
        # Set up mock for importlib
        mock_find_spec.return_value = MagicMock()
        
        # Mock ray
        ray_mock = MagicMock()
        ray_mock.is_initialized.return_value = True
        ray_mock.cluster_resources.return_value = {
            "GPU": 4,
            "CPU": 16,
            "node:worker1": 1,
            "node:worker2": 1
        }
        
        # Patch modules
        with patch.dict("sys.modules", {"ray": ray_mock}):
            result = get_ray_info()
        
        self.assertTrue(result["available"])
        self.assertEqual(result["gpus"], 4)
        self.assertEqual(result["cpus"], 16)
        self.assertEqual(result["nodes"], 2)
    
    @patch("importlib.util.find_spec")
    def test_ray_not_initialized(self, mock_find_spec):
        """Test Ray info when Ray is not initialized."""
        # Set up mock for importlib
        mock_find_spec.return_value = MagicMock()
        
        # Mock ray not initialized
        ray_mock = MagicMock()
        ray_mock.is_initialized.return_value = False
        
        # Patch modules
        with patch.dict("sys.modules", {"ray": ray_mock}):
            result = get_ray_info()
        
        self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()