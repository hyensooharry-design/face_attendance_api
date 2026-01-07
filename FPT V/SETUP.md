# 🚀 Quick Setup Guide

Follow these steps to get your face recognition system running:

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ This may take 5-10 minutes as it downloads PyTorch and other libraries.

## Step 2: Verify Installation

```bash
python -c "from models.face_models import load_mtcnn, load_facenet; print('✅ Installation successful!')"
```

> 📥 First run will download pre-trained models (~100MB)

## Step 3: Prepare Your Dataset

You have **two options**:

### Option A: Use Existing Images

1. Create folders in `dataset/` directory:
   ```
   dataset/
   ├── John/
   │   ├── photo1.jpg
   │   ├── photo2.jpg
   │   └── photo3.jpg
   └── Jane/
       ├── photo1.jpg
       └── photo2.jpg
   ```

2. Use 5-10 images per person with different angles and expressions

### Option B: Register with Webcam

```bash
python scripts/register_face.py
```

- Enter name when prompted
- Press **SPACE** to capture images
- Capture 10+ images with different angles

## Step 4: Build Face Database

```bash
# Create embeddings from images
python scripts/vectorize_faceset.py

# Build FAISS search index
python scripts/build_faiss_index.py
```

## Step 5: Verify Setup

```bash
python scripts/check_vectors.py
```

You should see:
- ✅ Number of registered people
- ✅ Embedding dimensions (512)
- ✅ FAISS index information

## Step 6: Run the System!

### Easy Menu (Recommended for beginners)
```bash
python main.py
```

### Streamlit Web Interface
```bash
streamlit run streamlit_app/app.py
```

Then open http://localhost:8501 in your browser

### OpenCV Standalone
```bash
python scripts/realtime_attendance.py
```

---

## 🎯 Quick Test Workflow

```bash
# 1. Install
pip install -r requirements.txt

# 2. Register a face
python scripts/register_face.py

# 3. Build database
python scripts/vectorize_faceset.py
python scripts/build_faiss_index.py

# 4. Run attendance
python scripts/realtime_attendance.py
```

---

## ⚠️ Common Issues

### Issue: "No module named 'torch'"
**Solution:** 
```bash
pip install torch torchvision
```

### Issue: "No module named 'facenet_pytorch'"
**Solution:**
```bash
pip install facenet-pytorch
```

### Issue: "FAISS index not found"
**Solution:** You need to build the index first:
```bash
python scripts/vectorize_faceset.py
python scripts/build_faiss_index.py
```

### Issue: "No face detected"
**Solution:**
- Ensure good lighting
- Face the camera directly
- Move closer to camera
- Check webcam is working

### Issue: "Could not open webcam"
**Solution:**
- Close other apps using webcam (Zoom, Teams, etc.)
- Try different camera index: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

---

## 📊 Expected Results

After setup, you should have:

```
FPT V/
├── dataset/           # Your face images
├── data/
│   ├── face_db.npy   # ✅ Generated embeddings
│   └── attendance.csv # ✅ Will be created on first run
├── faiss_index/
│   ├── face.index    # ✅ Generated FAISS index
│   └── names.npy     # ✅ Generated name mappings
└── ...
```

---

## 🎓 Next Steps

1. **Test Recognition**: Run the system and verify it recognizes faces
2. **Adjust Threshold**: If too many false positives, increase threshold to 0.7
3. **Add More People**: Use `register_face.py` to add more people
4. **View Attendance**: Check `data/attendance.csv` for records

---

## 💡 Tips for Best Results

- 📸 **More images = better accuracy** (10-20 per person)
- 💡 **Good lighting** is crucial
- 🔄 **Vary angles** during registration (front, left, right)
- 😊 **Different expressions** help (neutral, smiling)
- 👓 **Register with/without glasses** if you wear them
- 🎯 **Rebuild index** after adding new people

---

**Ready to start? Run:** `python main.py`
