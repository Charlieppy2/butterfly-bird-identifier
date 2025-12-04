# Butterfly & Bird Identifier

A deep learning-based web application for identifying butterflies and birds using AI-powered image classification.

## Project Overview

This is a web application system developed using deep learning technology to identify butterflies and birds from uploaded images. The system uses transfer learning technology, building a classification model based on MobileNetV2, capable of identifying 300+ species (200 bird species + 100+ butterfly/moth species), and provides a user-friendly web interface for image upload and identification.

## ✨ Key Features

### 🔍 Core Identification Features
- **Image Upload & Identification**: Support drag-and-drop or file selection (PNG, JPG, JPEG, GIF, WEBP)
- **Real-time Camera Capture**: Use device camera to capture images for identification
- **Batch Identification**: Upload multiple images at once for batch processing
- **Smart Recognition**: Deep learning-based image classification with Top-3 predictions and confidence scores

### 📊 Image Quality Analysis
- **Multi-dimensional Analysis**: Brightness, contrast, sharpness, saturation, resolution
- **Quality Score**: Overall quality score (0-100)
- **Smart Recommendations**: Provide improvement suggestions based on image quality issues

### 💬 AI Chat Assistant
- **Intelligent Q&A**: Answer questions about species identification, observation tips, etc.
- **Knowledge Base**: Contains information about habitats, observation times, photography tips
- **Trainable**: Support for expanding and training the AI assistant's knowledge base

### 📈 Statistical Analysis
- **Identification History Statistics**: Total identifications, unique species count, average confidence
- **Category Distribution**: Statistics for birds and butterflies/moths
- **Confidence Distribution**: Charts showing high/medium/low confidence distribution
- **Top Species**: Leaderboard of most frequently identified species
- **Time Distribution**: Trends in identification activity over time

### ❤️ Favorites Feature
- **Save Species**: One-click save for interesting identification results
- **Favorites Management**: View and manage all saved species
- **Data Persistence**: Use localStorage to save favorite data

### 📜 History Records
- **Identification History**: Automatically save recent identification records
- **Quick View**: Quickly browse historical identification results
- **Tab Switching**: Easy switching between history and favorites

## Tech Stack

### Model Training
- **TensorFlow/Keras**: Deep learning framework
- **MobileNetV2**: Pre-trained model (transfer learning)
- **Python 3.8+**: Programming language
- **OpenCV**: Image processing and quality analysis

### Web Application
- **Frontend**: React 18.2.0
  - Axios: HTTP client
  - Responsive design, mobile-friendly
- **Backend**: Flask 3.0.0
  - Flask-CORS: Cross-origin support
  - TensorFlow: Model inference
  - PIL/OpenCV: Image processing

## Project Structure

```
butterfly-bird-identifier/
├── data/
│   ├── raw/              # Raw dataset
│   ├── processed/        # Processed data (train/val/test)
│   └── dataset_info.txt  # Dataset information
├── models/
│   ├── training/         # Training scripts
│   │   ├── train_model.py      # Model training
│   │   ├── prepare_data.py     # Data preparation
│   │   ├── test_model.py       # Model testing
│   │   └── check_training.py   # Training progress check
│   └── trained/          # Trained models
│       ├── model.h5           # Trained model (using Git LFS)
│       └── class_names.json   # Class names list
├── web_app/
│   ├── frontend/         # React frontend application
│   │   ├── src/
│   │   │   ├── App.js         # Main application component
│   │   │   ├── App.css        # Stylesheet
│   │   │   └── index.js       # Entry file
│   │   ├── public/
│   │   │   └── index.html     # HTML template
│   │   └── package.json       # Frontend dependencies
│   ├── backend/          # Flask backend API
│   │   ├── app.py             # Flask application main file
│   │   ├── requirements.txt   # Python dependencies
│   │   ├── knowledge_base.json # AI assistant knowledge base
│   │   └── train_assistant.py  # AI assistant training script
│   └── preview.html      # Preview page
├── notebooks/            # Jupyter notebooks (data exploration)
├── report/              # Project reports
├── .gitattributes        # Git LFS configuration
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
   - Download: https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

2. **Node.js 16+**
   - Download: https://nodejs.org/
   - Recommended: LTS version

3. **Git LFS** (for downloading large files)
   ```bash
   git lfs install
   ```

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/Charlieppy2/butterfly-bird-identifier.git
cd butterfly-bird-identifier
```

#### 2. Install Backend Dependencies

```bash
cd web_app/backend
pip install -r requirements.txt
```

#### 3. Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

### Launch Application

#### Method 1: Manual Launch (Recommended)

**Start Backend Service:**

```bash
cd web_app/backend
python app.py
```

Backend service will start at `http://localhost:5000`

**Start Frontend Application:**

Open a new terminal window:

```bash
cd web_app/frontend
npm start
```

Frontend application will start at `http://localhost:3000`, browser will open automatically.

#### Method 2: Using Batch Files (Windows)

**Backend:**
```bash
cd web_app/backend
start_backend.bat
```

**Frontend:**
```bash
cd web_app/frontend
start_frontend.bat
```

## 📖 User Guide

### Identify Species

1. **Upload Image**:
   - Click "Choose File" button to select image
   - Or drag and drop image to upload area

2. **Capture Image**:
   - Click "📷 Use Camera" button
   - Allow browser to access camera
   - Click "📸 Capture" to take photo

3. **View Results**:
   - System will display identification results and confidence
   - Show Top-3 predictions
   - Automatically perform image quality analysis

### Use AI Assistant

1. Click the chat icon in the bottom right corner to open AI assistant
2. You can ask about:
   - Identification tips
   - Best observation times
   - Photography suggestions
   - Species information

### View Statistics

1. Click "📊 View Statistics" in the history section
2. View:
   - Total identifications
   - Category distribution (birds/butterflies)
   - Confidence distribution
   - Top identified species

### Favorites Feature

1. **Save Species**:
   - After identification, click the ❤️ button next to the result title

2. **View Favorites**:
   - Click "❤️ Favorites" tab
   - View all saved species

3. **Remove Favorites**:
   - Click "❌ Remove" button in favorites list
   - Or click ❤️ button again to unfavorite

## 🎓 Model Training

### Data Preparation

Organize raw images by category in `data/raw/` directory:

```
data/raw/
├── 001.Black_footed_Albatross/
│   ├── image1.jpg
│   └── ...
├── 002.Laysan_Albatross/
│   └── ...
└── ...
```

Run data preparation script:

```bash
cd models/training
python prepare_data.py
```

### Train Model

```bash
cd models/training
python train_model.py
```

Training parameters can be adjusted in `train_model.py`:
- `IMAGE_SIZE`: Image size (224, 224)
- `BATCH_SIZE`: Batch size (32)
- `EPOCHS`: Number of epochs (100)
- `LEARNING_RATE`: Learning rate (0.0001)

After training, model will be saved in `models/trained/model.h5`

### Check Training Progress

```bash
cd models/training
python check_training.py
```

### Test Model

```bash
cd models/training
python test_model.py
```

## 🤖 Train AI Assistant

For detailed guide, see: [如何訓練AI助手.md](如何訓練AI助手.md)

Quick start:

```bash
cd web_app/backend
python train_assistant.py
```

## 📊 Dataset Information

- **Total Classes**: 301 species (200 bird species + 101 butterfly/moth species)
- **Data Augmentation**: Rotation, flipping, scaling, brightness adjustment
- **Image Size**: 224x224
- **Train/Val/Test**: Automatically split

## 🔧 API Endpoints

### Backend API

- `GET /` - Health check
- `GET /api/health` - Model status
- `GET /api/classes` - Get all class names
- `POST /api/predict` - Image identification
- `POST /api/analyze-quality` - Image quality analysis
- `POST /api/statistics` - Get statistics
- `POST /api/chat` - AI chat assistant

## 🛠️ Development Environment

- Python 3.8+
- Node.js 16+
- TensorFlow 2.15.0+
- React 18.2.0
- Flask 3.0.0
- OpenCV 4.8.0+

## ⚠️ Notes

1. **Git LFS**: Model files use Git LFS storage, need to run `git lfs install` after cloning
2. **First Run**: First run needs to load model, may take some time
3. **GPU Acceleration**: Training model recommended to use GPU acceleration (Google Colab recommended)
4. **Disk Space**: Ensure sufficient disk space for dataset and model storage (model ~19MB)
5. **Browser Compatibility**: Recommended to use latest versions of Chrome, Firefox, or Edge

## 📝 Changelog

### v1.0.0 (Latest)
- ✨ Added favorites feature
- ✨ Added image quality analysis
- ✨ Added AI chat assistant
- ✨ Added identification history statistics and analysis
- ✨ Added batch identification mode
- 🐛 Fixed category distribution issue (butterflies correctly classified)
- 📦 Using Git LFS for large file management

## 📚 References

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [React Official Documentation](https://react.dev/)
- [Flask Official Documentation](https://flask.palletsprojects.com/)
- [Git LFS Documentation](https://git-lfs.github.com/)

## 📄 License

This project is for academic and educational purposes only.

## 👥 Contributing

Issues and Pull Requests are welcome!

## 📧 Contact

For questions or suggestions, please contact via GitHub Issues.

---

**Note**: Please ensure all necessary configurations and tests are completed before submission.

## 📄 Other Language Versions

- [繁體中文版 (Traditional Chinese)](README.zh-TW.md)
- [简体中文版 (Simplified Chinese)](README.zh-CN.md)
