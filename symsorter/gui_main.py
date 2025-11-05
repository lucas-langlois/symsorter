"""
Main entry point for SymSorter GUI application
This is used for creating the Windows executable
"""

import sys
from pathlib import Path

def main():
    """Launch the SymSorter GUI"""
    from PySide6.QtWidgets import QApplication
    from symsorter.image_browser import ImageBrowser
    
    app = QApplication(sys.argv)
    app.setApplicationName("SymSorter")
    app.setOrganizationName("SymSorter")
    
    # Create and show the main window
    browser = ImageBrowser()
    browser.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

