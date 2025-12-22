"""
Metal GPU Backend for Apple Silicon
====================================

Native Metal compute shaders for N-body simulation using Apple Silicon's
Unified Memory Architecture (UMA) for zero-copy CPU-GPU data sharing.
"""

from .metal_backend import MetalBarnesHutSimulation, is_metal_available

__all__ = ['MetalBarnesHutSimulation', 'is_metal_available']

