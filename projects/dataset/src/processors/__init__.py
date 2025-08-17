"""
Data processors package.

This package contains core data structures and processing utilities for
dataset integration and schema validation.
"""

from .unified_schema import CodeSample, QualityScores, DatasetSchema

__all__ = [
    'CodeSample',
    'QualityScores',
    'DatasetSchema'
]