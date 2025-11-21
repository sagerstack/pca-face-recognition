"""
PCA Face Recognition Demo - Main Streamlit Application

This is the main entry point for the multi-page Streamlit application that demonstrates
PCA-based face recognition from mathematical first principles.

Usage:
    poetry run streamlit run streamlit_app_clean.py
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to Python path for imports (using absolute path)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Initialize error logging immediately on import
try:
    # Change to app directory to ensure relative paths work
    os.chdir(current_dir)

    from utils.logger import get_error_logger, log_exception
    # Initialize logger immediately to create log file on startup
    logger = get_error_logger()
    logger.logger.info("Streamlit app initialization completed")

except ImportError:
    log_exception = None
    logger = None
except Exception:
    log_exception = None
    logger = None


def safe_page_exec(page_file_path: str, page_name: str):
    """
    Safely execute a page file with error handling.

    Args:
        page_file_path: Path to the page file
        page_name: Name of the page for logging
    """
    try:
        # Execute the page
        with open(page_file_path, 'r') as f:
            exec(f.read(), {})

    except Exception as e:
        # Log error silently
        if log_exception:
            log_exception(e, context=f"Loading page {page_name}")

        # Show user-friendly error message
        st.error(f"❌ Error loading {page_name}")
        st.error("Please check the log file for detailed information.")


def main():
    """Main application entry point with basic error handling."""
    try:
        # Set page configuration
        st.set_page_config(
            page_title="PCA Face Recognition Demo",
            page_icon="👤",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # App title and description
        st.title("👤 PCA Face Recognition Demo")
        st.markdown("Interactive demonstration of Principal Component Analysis for face recognition from mathematical first principles")

        # Multi-page app configuration
        PAGES = {
            "Eigenfaces": "src/pages/1_Eigenfaces.py",
            "Face Recognition": "src/pages/2_Face_Recognition.py",
            "Face Verification": "src/pages/3_Face_Verification.py"
        }

        # Page selection
        st.sidebar.title("Navigation")
        page_name = st.sidebar.selectbox("Select Page:", options=list(PAGES.keys()))

        # Load the selected page with error handling
        try:
            page_file_path = PAGES[page_name]

            if not os.path.exists(page_file_path):
                st.error(f"❌ Page file not found: {page_file_path}")
                st.info("Please check that all page files exist in the src/pages/ directory.")
            else:
                safe_page_exec(page_file_path, page_name)

        except KeyError:
            st.error(f"❌ Page not found in configuration: {page_name}")
            st.info("Available pages: " + ", ".join(PAGES.keys()))

        except Exception as e:
            # Log error silently
            if log_exception:
                log_exception(e, context=f"Loading page {page_name}")
            st.error(f"❌ Error loading page {page_name}")

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.info(
            "This demo implements PCA from first principles "
            "without using scikit-learn's PCA functionality. "
            "It demonstrates the mathematical concepts behind "
            "face recognition using eigenfaces."
        )

    except Exception as e:
        # Catch-all for any unhandled exceptions
        if log_exception:
            log_exception(e, context="Main application error")
        st.error("❌ An unexpected error occurred")
        st.error("Please check the log file for detailed information.")


if __name__ == "__main__":
    try:
        # Run the main app
        main()

    except KeyboardInterrupt:
        # Handle graceful shutdown
        pass

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if log_exception:
            log_exception(e, context="Fatal application error")
        sys.exit(1)