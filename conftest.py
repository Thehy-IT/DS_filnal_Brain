"""
conftest.py — Pytest configuration and shared fixtures for BrainTumorAI.

Located at project root so pytest can discover it automatically.
"""
import sys
import os

# Ensure the project root is on sys.path so `from src.xxx import` works
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
