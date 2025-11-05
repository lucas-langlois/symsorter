# SymSorter

**A CLIP-based image classification and similarity tool for intelligent image sorting**

SymSorter is a powerful image classification tool that leverages OpenAI's CLIP (Contrastive Language-Image Pre-training) and other vision models to help you sort, classify, and organize large collections of images based on semantic similarity and custom categories.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

## 🌟 Features

- **AI-Powered Image Encoding**: Uses CLIP, DINOv2, and other state-of-the-art vision models
- **GUI Embedding Generation**: Create embeddings directly from the GUI with real-time progress monitoring
- **Interactive GUI**: Fast, responsive interface for browsing and classifying thousands of images
- **Smart Similarity Sorting**: Double-click any image to sort by visual similarity
- **Custom Classification**: Define your own classes with custom keyboard shortcuts
- **YAML Configuration**: Flexible class definitions with descriptions and custom hotkeys
- **Batch Processing**: Efficient handling of large image collections
- **Caching System**: Smart caching for fast loading and smooth scrolling
- **Multiple Export Options**: Export as YOLO annotations or organize images into class folders
- **Thumbnail Scaling**: Multiple thumbnail sizes and crop options
- **Progress Tracking**: Save and resume classification work

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/lucas-langlois/symsorter.git
cd symsorter

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install package
pip install -e .
```

### Basic Usage

1. **Encode your images** (create embeddings):
   ```bash
   symsorter-encode /path/to/your/images --output embeddings.npz
   ```

2. **Create class definitions** (`classes.yaml`):
   ```yaml
   classes:
     - name: turtle
       keystroke: "5"
       description: "Sea turtle images"
     - name: waves
       keystroke: "2"
       description: "Wave and water surface images"
     - name: beach
       keystroke: "1"
       description: "Beach and shoreline images"
   ```

3. **Launch the GUI**:
   ```bash
   symsorter-gui --embeddings embeddings.npz --classes classes.yaml
   ```

4. **Start classifying**: Select images and press keyboard shortcuts to assign classes!

## 📋 Detailed Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for faster encoding)

### Step-by-Step Installation

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/NickMortimer/symsorter.git
   cd symsorter
   
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**:
   ```bash
   # Install all dependencies
   pip install -e .
   
   # Or install with development tools
   pip install -e ".[dev]"
   ```

3. **Verify Installation**:
   ```bash
   symsorter --help
   symsorter-gui --help
   symsorter-encode --help
   ```

## 📖 Usage Guide

### 1. Creating Image Embeddings

#### Option A: Using the GUI (Recommended)

The easiest way to generate embeddings is through the GUI:

1. Launch SymSorter GUI:
   ```bash
   symsorter-gui
   ```

2. Click **"Generate Embeddings"** button in the toolbar (or press **Ctrl+G**)

3. Configure your encoding:
   - **Input Directory**: Select folder containing your images
   - **Output Directory**: Where to save the .npz embeddings file
   - **Model Type**: Choose between CLIP, DINOv3, or DINOv3 TIMM
   - **Image Pattern**: File pattern to match (e.g., `*.jpg`, `*.png`)
   - **Recursive**: Enable to process subdirectories
   - **Batch Size**: Adjust for your GPU memory (default: 128)
   - **Crop Size**: Optional center crop before encoding

4. Click **"Start Encoding"** and monitor real-time progress

The embeddings will be saved to your specified output directory!

#### Option B: Using Command Line

You can also create embeddings from the command line:

```bash
# Basic encoding with CLIP
symsorter encode encode /path/to/images /path/to/output --pattern "*.jpg"

# Use DINOv3 model instead
symsorter encode encode /path/to/images /path/to/output --model-type dinov3

# Process with custom batch size and crop
symsorter encode encode /path/to/images /path/to/output --batch-size 32 --crop-size 512
```

**Supported Models:**
- `clip` (default): OpenAI CLIP ViT-B/32 - Fast, good for general images
- `dinov3`: Meta's DINOv3 vision transformer - Better for fine-grained details
- `dinov3_timm`: DINOv3 via timm library - Often faster than standard DINOv3

### 2. Setting Up Classification Classes

Create a `classes.yaml` file to define your classification categories:

```yaml
classes:
  - name: turtle
    keystroke: "5"
    description: "Sea turtle images"
  - name: waves
    keystroke: "2"
    description: "Wave and water surface images"
  - name: beach
    keystroke: "1"
    description: "Beach and shoreline images"
  - name: benthic
    keystroke: "3"
    description: "Seafloor and benthic habitat images"
```

**Alternative text format** (`classes.txt`):
```
turtle:5
waves:2
beach:1
benthic:3
```

### 3. Using the GUI

Launch the GUI with your embeddings and classes:

```bash
symsorter-gui --embeddings my_images.npz --classes classes.yaml
```

#### Temporal-Aware Sorting for Drone Surveys

For DJI drone surveys with 80% forward overlap, SymSorter can combine embedding similarity with temporal proximity to better group consecutive images of the same animal:

- **Enabled by default** for DJI images (with timestamps like `DJI_20250918073347_...`)
- **Toggle with Ctrl+T** or via View → Temporal-Aware Sorting menu
- **How it works**: When you double-click an image, it sorts by:
  - **70% embedding similarity** (visual content)
  - **30% temporal proximity** (capture time)
- **Result**: Images of the same animal from successive frames appear together

#### GUI Controls

**Navigation:**
- **Scroll**: Browse through images
- **Ctrl + Mouse Wheel**: Change thumbnail size
- **Ctrl + Plus/Minus**: Increase/decrease thumbnail size
- **Shift + Ctrl + Plus/Minus**: Adjust crop zoom

**Classification:**
- **Number Keys** or **Custom Keys**: Assign selected images to classes
- **Shift + F1-F12**: Default shortcuts for first 12 classes
- **Enter**: Assign to last used class
- **Double-click image**: Sort by similarity (temporal-aware for DJI images)

**Sorting:**
- **Ctrl + T**: Toggle temporal-aware sorting (for DJI drone images with 80% overlap)

**File Operations:**
- **Ctrl + G**: Generate embeddings from images
- **Ctrl + O**: Load embeddings
- **Ctrl + S**: Save classifications
- **Ctrl + E**: Export YOLO annotations
- **Ctrl + Shift + E**: Export images to folders by class
- **R**: Reset image order

**Filtering:**
- Use the dropdown to filter by:
  - All Images
  - Unallocated (unclassified)
  - Specific class names

### 4. Working with Classifications

**Select Multiple Images:**
- Click and drag to select multiple images
- Hold Ctrl and click to add individual images to selection
- Hold Shift and click to select ranges

**Assign Classes:**
- Select images and press the assigned key (e.g., "5" for turtle)
- Images are automatically hidden from "Unallocated" view
- Switch to "All Images" view to see classified images

**Save Progress:**
- Press Ctrl+S or use File → Save Classifications
- Classifications are saved back to the NPZ file
- Resume work by loading the same NPZ file

### 5. Export Results

Export your classifications for use in other tools:

#### Export YOLO Annotations
```bash
# Export YOLO format annotations
# Use File → Export YOLO Annotations in GUI
# Creates labels/ directory with .txt files for each image
```

#### Export Images to Folders by Class

Organize your classified images into separate folders for easy access:

1. **Press Ctrl+Shift+E** or use **File → Export Images to Folders...**
2. **Select export directory** where class folders will be created
3. **Choose operation**:
   - **Copy**: Keeps original images in place, creates copies in class folders
   - **Move**: Moves images to class folders (originals are deleted)
4. **Optional CSV Filtering**: If `patch_based_predictions.csv` is found in the source folder, you'll be asked if you want to:
   - Filter the CSV to only include classified detections
   - Update image paths to point to the new exported locations
   - Save as `filtered_predictions.csv` in the export directory
5. Images are automatically organized:
   ```
   export_directory/
   ├── turtle/
   │   ├── image001.jpg
   │   └── image023.jpg
   ├── ray/
   │   ├── image005.jpg
   │   └── image012.jpg
   ├── dugong/
   │   └── image030.jpg
   └── filtered_predictions.csv  (if CSV filtering was enabled)
   ```

**Benefits:**
- Easily share classified images with colleagues
- Prepare datasets for training or analysis
- Archive organized collections
- Duplicate filenames are automatically handled
- Automatically filter and update detection CSV files
- Keep your data organized and ready for downstream processing

## ⚙️ Configuration

### Environment Variables

```bash
# Set default device for encoding
export SYMSORTER_DEVICE=cuda

# Set cache directory
export SYMSORTER_CACHE_DIR=/path/to/cache
```

### Performance Tuning

**For Large Collections (10k+ images):**
- Use a fast SSD for image storage
- Increase batch size for encoding: `--batch-size 64`
- Use CUDA if available: `--device cuda`

**For Better GUI Performance:**
- Use smaller thumbnail sizes for very large collections
- Enable image caching (enabled by default)
- Close other applications to free up memory

## 🔧 Advanced Usage

### Command Line Interface

```bash
# Encode with specific model and settings
symsorter-encode images/ \
  --model dinov3 \
  --batch-size 32 \
  --device cuda \
  --output marine_survey.npz

# Launch GUI with specific settings
symsorter-gui \
  --embeddings marine_survey.npz \
  --classes marine_classes.yaml
```

### Integration with Other Tools

SymSorter works well with:
- **Computer Vision Pipelines**: Export embeddings for clustering/analysis
- **YOLO Training**: Export annotations for object detection training
- **Data Science Workflows**: Load embeddings as numpy arrays for analysis

### Programmatic Usage

```python
from symsorter.clip_encode import load_existing_embeddings
import numpy as np

# Load embeddings
embeddings = load_existing_embeddings('my_images.npz')

# Access embedding vectors
for filename, embedding in embeddings.items():
    print(f"{filename}: {embedding.shape}")
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black symsorter/
isort symsorter/

# Type checking
mypy symsorter/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the CLIP model
- **Meta AI** for the DINOv3 model
- **Anthropic** for development assistance
- **Qt/PySide6** for the GUI framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/lucas-langlois/symsorter/issues)
- **Documentation**: [Project Wiki](https://github.com/lucas-langlois/symsorter/wiki)

## 🗺️ Roadmap

- [x] **GUI Embedding Generation** - Generate embeddings directly from the interface
- [ ] Additional vision model support
- [ ] Batch classification suggestions
- [ ] Integration with cloud storage
- [ ] Advanced similarity metrics
- [ ] Multi-user collaboration features

---

**SymSorter** - Making image classification intelligent and efficient! 🖼️✨
