# Building SymSorter Windows Executable

This guide explains how to create a standalone Windows executable (.exe) for SymSorter that can be distributed to users who don't have Python installed.

## Prerequisites

1. **Windows OS** (for building Windows executables)
2. **Conda environment with SymSorter installed**
3. **PyInstaller** (will be installed in the build process)

## Quick Build

### Option 1: Automated Build (Recommended)

Simply run the batch file:
```bash
build_exe.bat
```

This will:
- Install PyInstaller if needed
- Build the executable
- Create a `dist/` folder with `SymSorter.exe`

### Option 2: Manual Build

1. **Activate your conda environment:**
   ```bash
   conda activate symsorter
   ```

2. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

3. **Run the build script:**
   ```bash
   python build_exe.py
   ```

4. **Find your executable:**
   ```
   dist/SymSorter.exe
   ```

## What Gets Created

The build process creates:
- `dist/SymSorter.exe` - The standalone executable (1-2 GB due to PyTorch)
- `build/` - Temporary build files (can be deleted)
- `SymSorter.spec` - PyInstaller specification file (for advanced customization)

## Distributing the Executable

### For Users Without Python/Conda:

1. **Simple Distribution:**
   - Just send them `SymSorter.exe` from the `dist/` folder
   - They can double-click it to run
   - First launch may take 30-60 seconds (unpacking)

2. **Installer Package (Optional):**
   - Use Inno Setup or NSIS to create a proper installer
   - Adds desktop shortcuts and Start menu entries
   - See `create_installer.md` for details

### Important Notes:

⚠️ **File Size:** The executable will be 1-2 GB due to PyTorch and deep learning libraries. This is normal.

⚠️ **Antivirus:** Some antivirus software may flag PyInstaller executables. Users may need to add an exception.

⚠️ **First Run:** The first time the exe is run, Windows may show a SmartScreen warning. Users should click "More info" → "Run anyway".

⚠️ **GPU Support:** The executable includes CUDA support if your build machine has PyTorch with CUDA. For CPU-only, it will still work but be slower.

## Troubleshooting

### Build fails with "module not found"
- Make sure all dependencies are installed: `pip install -e .`
- Check that you're in the correct conda environment

### Executable won't start
- Try running from command line to see error messages
- Check Windows Event Viewer for crash details
- Ensure all hidden imports are listed in `build_exe.py`

### Executable is too large
- The size is mainly from PyTorch (~1.5 GB)
- You can use UPX compression (add `--upx-dir` to PyInstaller args)
- Consider creating a --onedir build instead of --onefile

### Missing torch/CUDA errors
- Add `--collect-all torch` to PyInstaller arguments
- Ensure PyTorch is properly installed in your environment

## Advanced: Customizing the Build

Edit `build_exe.py` to:
- Change the executable name
- Add/remove hidden imports
- Modify the icon
- Switch between `--onefile` and `--onedir`
- Add data files or resources

## Creating an Installer

For a more professional distribution, create an installer with Inno Setup:

1. Download and install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Use the `symsorter_installer.iss` script (see `create_installer.md`)
3. Compile to create `SymSorter_Setup.exe`

This creates a proper Windows installer with:
- Installation wizard
- Desktop shortcut creation
- Start menu entry
- Uninstaller
- Version information

## Support

If you encounter issues building the executable:
1. Check the [GitHub Issues](https://github.com/lucas-langlois/symsorter/issues)
2. Ensure your environment has all dependencies: `pip install -e .`
3. Try a clean build: Delete `build/` and `dist/` folders and rebuild

