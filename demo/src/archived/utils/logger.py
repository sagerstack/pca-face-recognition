"""
Error Logging and Debugging Utilities for PCA Face Recognition App

This module provides comprehensive error logging and exception handling
capabilities for the Streamlit application. It includes utilities for:

- Setting up detailed logging configuration
- Catching and logging all exceptions
- Performance monitoring
- Debugging utilities for troubleshooting

Author: PCA Face Recognition Team
"""

import logging
import sys
import traceback
import os
import datetime
import functools
import json
from typing import Optional, Any, Callable
import inspect
from pathlib import Path


class DetailedFormatter(logging.Formatter):
    """
    Custom formatter that includes detailed context information:
    PID, filename, class, and function name for each log message.
    """

    def __init__(self):
        super().__init__()
        self.pid = os.getpid()

    def get_caller_info(self):
        """Extract detailed caller information from the call stack."""
        # Get the call stack and skip the first few frames (logging internals)
        stack = inspect.stack()

        # Skip logging-related frames
        start_frame = 3
        for i in range(start_frame, len(stack)):
            frame = stack[i]
            frame_info = frame[0]

            # Skip frames that are in the logging system itself
            filename = os.path.basename(frame_info.f_code.co_filename)
            if not any(skip_name in filename for skip_name in ['error_logger.py', 'logging/__init__.py']):

                # Extract file, class, and function information
                file_path = frame_info.f_code.co_filename
                file_name = os.path.basename(file_path)
                function_name = frame_info.f_code.co_name

                # Get class name if this is a method call
                class_name = ""
                if 'self' in frame_info.f_locals:
                    class_name = frame_info.f_locals['self'].__class__.__name__
                elif 'cls' in frame_info.f_locals:
                    class_name = frame_info.f_locals['cls'].__name__

                # Extract line number
                line_number = frame_info.f_lineno

                return {
                    'pid': self.pid,
                    'file': file_name,
                    'function': function_name,
                    'class': class_name,
                    'line': line_number,
                    'full_path': file_path
                }

        # Fallback if we can't find a suitable frame
        return {
            'pid': self.pid,
            'file': 'unknown',
            'function': 'unknown',
            'class': '',
            'line': 0,
            'full_path': 'unknown'
        }

    def format(self, record):
        """Format the log record with detailed context information."""
        caller_info = self.get_caller_info()

        # Build the context string
        context_parts = [
            f"PID:{caller_info['pid']}",
        ]

        # Add file information
        if caller_info['file'] != 'unknown':
            context_parts.append(f"File:{caller_info['file']}")

        # Add class information if available
        if caller_info['class']:
            context_parts.append(f"Class:{caller_info['class']}")

        # Add function information
        if caller_info['function'] != 'unknown':
            context_parts.append(f"Func:{caller_info['function']}")

        # Combine context parts
        context_str = " | ".join(context_parts)

        # Format the final message
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S,%f')[:-3]
        level = record.levelname.upper().ljust(8)

        return f"{timestamp} [{level}] {context_str} | {record.getMessage()}"


class StreamlitErrorLogger:
    """
    Simplified error logging system for Streamlit applications.

    This class provides methods to catch, log, and handle errors that occur
    in the Streamlit application with minimal output to the frontend.
    """

    def __init__(self, debug_mode: bool = False):
        """
        Initialize the error logger.

        Args:
            debug_mode: Enable detailed debug logging
        """
        self.debug_mode = debug_mode

        # Generate timestamped log filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = f"logs/streamlit-log-{timestamp}.log"

        self.setup_logging()

    def setup_logging(self):
        """Set up detailed logging configuration with context information."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Get logger for our app
        self.logger = logging.getLogger('pca_face_recognition')
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)

        # Prevent propagation to root logger to avoid duplicates
        self.logger.propagate = False

        # Ensure only one file handler exists
        if len(self.logger.handlers) == 0:
            file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(DetailedFormatter())
            self.logger.addHandler(file_handler)

        self.logger.info("=== PCA Face Recognition Application Started ===")

    def log_exception(self, exc: Exception, context: Optional[str] = None,
                      extra_data: Optional[dict] = None):
        """
        Log an exception with detailed context information.

        Args:
            exc: The exception to log
            context: Additional context about where the error occurred
            extra_data: Additional data to include in the log
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Build detailed error message with context
        context_part = f" | Context: {context}" if context else ""
        exc_name = exc_type.__name__ if exc_type else 'Unknown'

        error_msg = f"Exception: {exc_name} | Message: {str(exc)}{context_part}"

        # Add extra data if provided
        if extra_data:
            error_msg += f" | Extra Data: {json.dumps(extra_data)}"

        # Log the exception with enhanced context
        self.logger.error(error_msg)

        # Log full traceback in debug mode
        if self.debug_mode:
            traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            self.logger.error(f"Full Traceback:\n{traceback_str}")

        return {
            'exception_type': exc_type.__name__ if exc_type else 'Unknown',
            'exception_message': str(exc),
            'context': context,
            'extra_data': extra_data
        }

    def safe_execute(self, func: Callable, *args, context: str = None,
                default_return: Any = None, **kwargs) -> Any:
        """
        Safely execute a function and catch all exceptions.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            context: Context for error logging
            default_return: Value to return if function fails
            **kwargs: Keyword arguments for the function

        Returns:
            Function result or default_return if exception occurs
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log_exception(e, context=context or f"safe_execute({func.__name__})")
            if default_return is not None:
                return default_return
            raise


# Global error logger instance
_global_logger = None

def get_error_logger() -> StreamlitErrorLogger:
    """Get the global error logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StreamlitErrorLogger()
    return _global_logger

def force_new_logger() -> StreamlitErrorLogger:
    """Force creation of a new logger instance."""
    global _global_logger
    _global_logger = StreamlitErrorLogger()
    return _global_logger


def log_exception(exc: Exception, context: Optional[str] = None, extra_data: Optional[dict] = None):
    """Convenience function to log an exception using the global logger."""
    logger = get_error_logger()
    return logger.log_exception(exc, context, extra_data)


def safe_streamlit_execute(func: Callable, *args, context: str = None,
                          default_return: Any = None, **kwargs) -> Any:
    """
    Convenience function to safely execute Streamlit functions.

    Args:
        func: Function to execute
        *args: Positional arguments
        context: Context for error logging
        default_return: Value to return if function fails
        **kwargs: Keyword arguments

    Returns:
        Function result or default_return
    """
    logger = get_error_logger()
    return logger.safe_execute(func, *args, context=context, default_return=default_return, **kwargs)


def log_streamlit_error(message: str, context: Optional[str] = None,
                       extra_data: Optional[dict] = None):
    """
    Log a Streamlit error message and return the message for display.

    Args:
        message: Error message to log and display
        context: Additional context about where the error occurred
        extra_data: Additional data to include in the log

    Returns:
        The same message for use with st.error()
    """
    logger = get_error_logger()
    full_message = f"Streamlit UI Error: {message}"
    logger.logger.error(full_message)
    return message


def log_streamlit_warning(message: str, context: Optional[str] = None,
                         extra_data: Optional[dict] = None):
    """
    Log a Streamlit warning message and return the message for display.

    Args:
        message: Warning message to log and display
        context: Additional context about where the warning occurred
        extra_data: Additional data to include in the log

    Returns:
        The same message for use with st.warning()
    """
    logger = get_error_logger()
    full_message = f"Streamlit UI Warning: {message}"
    logger.logger.warning(full_message)
    return message


def log_streamlit_info(message: str, context: Optional[str] = None,
                      extra_data: Optional[dict] = None):
    """
    Log a Streamlit info message and return the message for display.

    Args:
        message: Info message to log and display
        context: Additional context about where the info occurred
        extra_data: Additional data to include in the log

    Returns:
        The same message for use with st.info()
    """
    logger = get_error_logger()
    full_message = f"Streamlit UI Info: {message}"
    logger.logger.info(full_message)
    return message


def log_streamlit_success(message: str, context: Optional[str] = None,
                         extra_data: Optional[dict] = None):
    """
    Log a Streamlit success message and return the message for display.

    Args:
        message: Success message to log and display
        context: Additional context about where the success occurred
        extra_data: Additional data to include in the log

    Returns:
        The same message for use with st.success()
    """
    logger = get_error_logger()
    full_message = f"Streamlit UI Success: {message}"
    logger.logger.info(full_message)
    return message


# Streamlit UI message wrapper functions
def st_error_logged(message: str, context: Optional[str] = None,
                   extra_data: Optional[dict] = None):
    """
    Display st.error message and log it automatically.

    Args:
        message: Error message to display and log
        context: Additional context for logging
        extra_data: Additional data to include in the log
    """
    import streamlit as st
    log_streamlit_error(message, context, extra_data)
    st.error(message)


def st_warning_logged(message: str, context: Optional[str] = None,
                     extra_data: Optional[dict] = None):
    """
    Display st.warning message and log it automatically.

    Args:
        message: Warning message to display and log
        context: Additional context for logging
        extra_data: Additional data to include in the log
    """
    import streamlit as st
    log_streamlit_warning(message, context, extra_data)
    st.warning(message)


def st_info_logged(message: str, context: Optional[str] = None,
                  extra_data: Optional[dict] = None):
    """
    Display st.info message and log it automatically.

    Args:
        message: Info message to display and log
        context: Additional context for logging
        extra_data: Additional data to include in the log
    """
    import streamlit as st
    log_streamlit_info(message, context, extra_data)
    st.info(message)


def st_success_logged(message: str, context: Optional[str] = None,
                     extra_data: Optional[dict] = None):
    """
    Display st.success message and log it automatically.

    Args:
        message: Success message to display and log
        context: Additional context for logging
        extra_data: Additional data to include in the log
    """
    import streamlit as st
    log_streamlit_success(message, context, extra_data)
    st.success(message)


def main():
    """Test the error logging system."""
    print("Testing Streamlit Error Logger...")

    logger = get_error_logger()

    # Test exception logging
    try:
        1 / 0
    except Exception as e:
        error_info = logger.log_exception(e, context="Test division by zero")
        print(f"Logged exception: {error_info['exception_type']}")

    # Test safe execute
    def failing_function():
        raise ValueError("This function always fails")

    safe_result = logger.safe_execute(failing_function, default_return="safe_result")
    print(f"Safe execute result: {safe_result}")

    print(f"Log file created at: {logger.log_file}")
    print("✅ Error logger test complete!")


if __name__ == "__main__":
    main()