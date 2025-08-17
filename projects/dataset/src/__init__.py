"""
CodeQual Dataset Creation Module

This module handles the creation and enhancement of the CodeQual dataset,
including integration of multiple data sources and analysis capabilities.
"""

from .processors import CodeSample, QualityScores, DatasetSchema
from .integrators import DataSourceIntegrator, RawDataSample, DataSourceRegistry, CodeEvalIntegrator

__version__ = "2.0.0"
__all__ = [
    "CodeSample",
    "QualityScores", 
    "DatasetSchema",
    "DataSourceIntegrator",
    "RawDataSample",
    "DataSourceRegistry",
    "CodeEvalIntegrator"
]