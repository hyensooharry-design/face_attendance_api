# 🚀 FPS Optimization Guide

## Current Optimizations Applied ✅

### 1. **Frame Skipping** (FRAME_SKIP = 5)
- Face recognition runs every 5th frame
- Display updates every frame (smooth video)
- **Expected FPS**: 20-30 FPS

### 2. **Frame Resizing** (320x240)
- Input frame resized before face detection
- 4x less pixels to process = 2-3x faster
- **Speed improvement**: ~50-70%

### 3. **Camera Resolution** (640x480)
- Lower resolution = faster capture
- Still good enough for face recognition

---

## 📊 Expected Performance

| Configuration | FPS | Recognition Frequency |
|--------------|-----|----------------------|
| **Current (Optimized)** | 20-30 | Every 5 frames (~6 times/sec) |
| Original | 6-10 | Every frame |

---

## 🎛️ Tuning Options

### If FPS is still low, adjust in `app.py`:

```python
# Line 87-88
FRAME_SKIP = 10  # Try 7, 10, or 15 for even higher FPS
```

**Trade-off:**
- Higher FRAME_SKIP = Faster FPS, but less frequent recognition
- Lower FRAME_SKIP = More accurate, but slower FPS

### Frame Size Options:

```python
# Line 63 in recognize_face()
small_frame = cv2.resize(frame, (320, 240))  # Current
# OR
small_frame = cv2.resize(frame, (240, 180))  # Even faster
# OR
small_frame = cv2.resize(frame, (160, 120))  # Maximum speed (may reduce accuracy)
```

---

## 🔥 Why CPU is Slow

**The bottleneck:**
1. **MTCNN** (face detection): ~30-50ms
2. **FaceNet** (embedding): ~50-100ms
3. **Total**: 80-150ms per recognition

**With optimizations:**
- Resizing: Reduces MTCNN time by 50-70%
- Frame skipping: Only runs every 5th frame
- Result: Smooth 20-30 FPS display

---

## 💡 Additional Tips

### For Maximum Speed:
```python
TARGET_FPS = 30
FRAME_SKIP = 10
small_frame = cv2.resize(frame, (240, 180))
```

### For Maximum Accuracy:
```python
TARGET_FPS = 15
FRAME_SKIP = 3
small_frame = cv2.resize(frame, (480, 360))
```

### Balanced (Recommended):
```python
TARGET_FPS = 30
FRAME_SKIP = 5
small_frame = cv2.resize(frame, (320, 240))
```

---

## 🎯 Current Settings (Already Applied)

✅ `TARGET_FPS = 30`
✅ `FRAME_SKIP = 5`
✅ `Frame resize to 320x240`
✅ `Camera resolution 640x480`

**Expected Result**: 20-30 FPS with face recognition every 5 frames

---

## 🚀 Test It Now!

```bash
streamlit run streamlit_app/app.py
```

You should see the FPS counter in the top-right corner showing **20-30 FPS**!
