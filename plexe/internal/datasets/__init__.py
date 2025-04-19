"""
Application entry point for the data generation service.

The data generation service is an internal API that generates synthetic data that is meant to capture a particular
data distribution, either with data or without data (low-data regime). The service also exposes functionality for
validating the synthetic data against real data, if available.
"""

from .config import config as config
