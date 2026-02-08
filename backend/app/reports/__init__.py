"""
Reports module for generating PDF reports from simulation data.
"""

from .pdf_generator import RegulatoryReportGenerator
from .templates import BASEL_III_TEMPLATE, STRESS_TEST_TEMPLATE, SYSTEMIC_RISK_TEMPLATE

__all__ = [
    "RegulatoryReportGenerator",
    "BASEL_III_TEMPLATE",
    "STRESS_TEST_TEMPLATE",
    "SYSTEMIC_RISK_TEMPLATE",
]
