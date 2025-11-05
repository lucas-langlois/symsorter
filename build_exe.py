"""
Build script for creating Windows executable of SymSorter GUI
Uses PyInstaller to package the application with all dependencies
"""

import PyInstaller.__main__
import sys
import os
from pathlib import Path

def build_exe():
    """Build the executable using PyInstaller"""
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # PyInstaller arguments
    args = [
        'symsorter/gui_main.py',  # Entry point script
        '--name=SymSorter',  # Name of the executable
        '--onefile',  # Create a single executable file
        '--windowed',  # Don't show console window
        '--icon=symsorter/resources/icon.ico' if (current_dir / 'symsorter' / 'resources' / 'icon.ico').exists() else '',
        
        # Add hidden imports for deep learning and GUI libraries
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=transformers',
        '--hidden-import=timm',
        '--hidden-import=clip',
        '--hidden-import=PIL',
        '--hidden-import=numpy',
        '--hidden-import=sklearn',
        '--hidden-import=pandas',
        '--hidden-import=PySide6',
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=PySide6.QtWidgets',
        
        # Collect data files
        '--collect-all=torch',
        '--collect-all=torchvision',
        '--collect-all=transformers',
        '--collect-all=timm',
        
        # Add symsorter package
        '--add-data=symsorter;symsorter',
        
        # Exclude unnecessary packages to reduce size
        '--exclude-module=matplotlib',
        '--exclude-module=IPython',
        '--exclude-module=notebook',
        
        # Other options
        '--clean',  # Clean PyInstaller cache
        '--noconfirm',  # Replace output directory without asking
        
        # Output directory
        '--distpath=dist',
        '--workpath=build',
        '--specpath=build',
    ]
    
    # Remove empty icon argument if icon doesn't exist
    args = [arg for arg in args if arg]
    
    print("Building SymSorter executable...")
    print(f"Arguments: {' '.join(args)}")
    
    try:
        PyInstaller.__main__.run(args)
        print("\n" + "="*60)
        print("Build completed successfully!")
        print(f"Executable location: {current_dir / 'dist' / 'SymSorter.exe'}")
        print("="*60)
    except Exception as e:
        print(f"\nBuild failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    build_exe()

