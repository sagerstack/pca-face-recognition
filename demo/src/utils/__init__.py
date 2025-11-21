"""
Utility Functions Module

This module contains general utility functions, Streamlit helpers, and file
handling utilities for the PCA face recognition system.

Classes:
- StreamlitUtils: Streamlit-specific UI components and utilities
- FileUtils: File handling and I/O utilities
"""

from .logger import (
    get_error_logger,
    log_exception,
    safe_streamlit_execute,
    st_error_logged,
    st_warning_logged,
    st_info_logged,
    st_success_logged,
    log_streamlit_error,
    log_streamlit_warning,
    log_streamlit_info,
    log_streamlit_success
)
# from .streamlit_utils import StreamlitUtils  # Not implemented yet
# from .file_utils import FileUtils  # Not implemented yet

__all__ = [
    "get_error_logger",
    "log_exception",
    "safe_streamlit_execute",
    "st_error_logged",
    "st_warning_logged",
    "st_info_logged",
    "st_success_logged",
    "log_streamlit_error",
    "log_streamlit_warning",
    "log_streamlit_info",
    "log_streamlit_success"
]