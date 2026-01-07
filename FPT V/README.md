# 🎓 Face Recognition Attendance System

A real-time face recognition attendance system using **MTCNN** for face detection, **FaceNet** for embeddings, and **FAISS** for efficient similarity search.

## 🌟 Features

- ✅ Real-time face detection and recognition
- ✅ Automatic attendance logging (IN/OUT toggle)
- ✅ FAISS-based fast similarity search
- ✅ Streamlit web interface
- ✅ Standalone OpenCV interface
- ✅ Easy face registration system
- ✅ CPU-optimized (no GPU required)

## 📋 Requirements

- Python 3.8 or higher
- Webcam

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Face Dataset

Create a `dataset` directory with subdirectories for each person:

```
dataset/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

**OR** use the registration tool:

```bash
python scripts/register_face.py
```

### 3. Build Face Database

```bash
# Step 1: Create face embeddings
python scripts/vectorize_faceset.py

# Step 2: Build FAISS index
python scripts/build_faiss_index.py
```

### 4. Run the System

**Option A: Menu Interface (Recommended)**
```bash
python main.py
```

**Option B: Streamlit Web App**
```bash
streamlit run streamlit_app/app.py
```

**Option C: OpenCV Standalone**
```bash
python scripts/realtime_attendance.py
```

## 📁 Project Structure

```
FPT V/
├── models/
│   └── face_models.py          # MTCNN & FaceNet loaders
├── utils/
│   ├── faiss_utils.py          # FAISS index management
│   ├── attendance_utils.py     # Attendance logging
│   └── face_utils.py           # Face processing utilities
├── scripts/
│   ├── register_face.py        # Register new faces
│   ├── vectorize_faceset.py    # Create embeddings
│   ├── build_faiss_index.py    # Build FAISS index
│   ├── check_vectors.py        # Verify embeddings
│   └── realtime_attendance.py  # OpenCV attendance
├── streamlit_app/
│   └── app.py                  # Streamlit web interface
├── dataset/                     # Face images (organized by person)
├── data/                        # Generated data files
│   ├── face_db.npy             # Face embeddings
│   └── attendance.csv          # Attendance records
├── faiss_index/                # FAISS index files
│   ├── face.index              # FAISS index
│   └── names.npy               # Name mappings
├── main.py                      # Main menu interface
└── requirements.txt            # Python dependencies
```

## 🔧 Usage Guide

### Registering New Faces

1. Run the registration script:
   ```bash
   python scripts/register_face.py
   ```

2. Enter the person's name

3. Press **SPACE** to capture images (10 recommended)

4. Press **ESC** to cancel

5. Rebuild the database:
   ```bash
   python scripts/vectorize_faceset.py
   python scripts/build_faiss_index.py
   ```

### Checking System Status

```bash
python scripts/check_vectors.py
```

This displays:
- Number of registered people
- Embedding dimensions
- FAISS index statistics

### Viewing Attendance Records

Attendance is logged to `data/attendance.csv` with columns:
- **Name**: Person's name
- **Date**: Date (YYYY-MM-DD)
- **Time**: Time (HH:MM:SS)
- **Status**: IN or OUT
- **Confidence**: Recognition confidence (0-1)

## ⚙️ Configuration

### Face Recognition Threshold

Edit the threshold in the respective files:

**Streamlit App** (`streamlit_app/app.py`):
```python
name, score = recognize_face(frame, threshold=0.6)  # Adjust 0.6
```

**OpenCV App** (`scripts/realtime_attendance.py`):
```python
name, confidence = recognize_face(frame, mtcnn, facenet, index, names, threshold=0.6)
```

- **Higher threshold** (0.7-0.8): More strict, fewer false positives
- **Lower threshold** (0.4-0.5): More lenient, may have false positives

### Attendance Cooldown

To prevent duplicate logging, there's a cooldown period in `realtime_attendance.py`:

```python
cooldown_seconds = 5  # Adjust as needed
```

## 🐛 Troubleshooting

### "No module named 'facenet_pytorch'"

```bash
pip install facenet-pytorch
```

### "FAISS index not found"

Run the setup commands:
```bash
python scripts/vectorize_faceset.py
python scripts/build_faiss_index.py
```

### "Could not open webcam"

- Check if another application is using the webcam
- Try changing camera index in code: `cv2.VideoCapture(1)` instead of `0`

### Low recognition accuracy

- Capture more images per person (15-20)
- Ensure good lighting conditions
- Vary face angles during registration
- Lower the recognition threshold

### "No face detected"

- Ensure proper lighting
- Face the camera directly
- Move closer to the camera
- Check if MTCNN is loaded correctly

## 📊 Performance

- **Face Detection**: ~30-50ms per frame (CPU)
- **Face Recognition**: ~50-100ms per frame (CPU)
- **FAISS Search**: <1ms for 100 people
- **Memory Usage**: ~500MB (models + index)

## 🔒 Privacy & Security

- All processing is done **locally** on your machine
- No data is sent to external servers
- Face embeddings are stored as numerical vectors (not images)
- Attendance data is stored in local CSV files

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **facenet-pytorch**: Pre-trained MTCNN and FaceNet models
- **FAISS**: Efficient similarity search by Facebook AI
- **Streamlit**: Easy web app framework

## 📧 Support

For issues or questions, please check:
1. This README
2. Error messages in console
3. Verify all dependencies are installed

---

**Made with ❤️ for attendance automation**
