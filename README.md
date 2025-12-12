# Art Style Classification and Period Detection
A deep learning system that automatically classifies paintings by artistic movement using ResNet50 transfer learning on the WikiArt dataset.

### Dataset
* Source: WikiArt dataset
* Size: 81,444 images
* Classes: 27 art styles/movements
* Note: Dataset is ~15-20GB. Ensure adequate storage space.

### Environment Specifications
#### Hardware Used
CPU: Intel Core i9-14900KF
GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)
RAM: 64GB DDR5
Storage: 2TB SSD
#### Software Requirements
Python 3.8+
CUDA-capable GPU (recommended)
20GB+ free disk space

### Installation
1. Clone or download this project
2. Install required packages:
  * pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
  * For GPU support (Windows):
  * pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
3. Download WikiArt dataset
  * Organize images in folders by art style
  * Each folder name = class label

### Folder Structure
project/
├── finalproject_art_classification.ipynb    
├── README.md                                
├── WikiArt/                                 
│   ├── Impressionism/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── Renaissance/
│   │   └── ...
│   └── ... (27 art style folders)
├── resnet50_art_classifier.pth              
└── confusion_matrix.png                     

### How to Run
1. Update the data path in the notebook:
data_dir = "C:/Projects/WikiArt"  # Change to your dataset location
2. Open the notebook:
jupyter notebook finalproject_art_classification.ipynb
3. Run all cells in order:
Cell 1: Imports and GPU check
Cell 2: Configuration
Cell 3: Data loading
Cell 4: Model setup
Cell 5: Training (~45 min with GPU, 2-3 hours with CPU)
Cell 6: Evaluation
4. View results:
Metrics printed in notebook
Confusion matrix saved as confusion_matrix.png
Model saved as resnet50_art_classifier.pth

### Training Time
With GPU (RTX 4090): ~1 hour and 20 minutes for 15 epochs
First epoch: 10-15 minutes (data loading initialization)
Subsequent epochs: 3-8 minutes each

### Results
Accuracy: 55.26%
Macro Precision: 61.18%
Macro Recall: 49.60%
Macro F1-Score: 53.14%
Top-5 Accuracy: 92.14%

### Notes
Large dataset (81k images) requires significant RAM during loading
GPU highly recommended for reasonable training time
num_workers=4 optimizes data loading speed
Model uses transfer learning from ImageNet pre-trained ResNet50
