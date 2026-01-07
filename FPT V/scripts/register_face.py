"""
Register a new face to the system.
Captures images from webcam and saves to dataset.
"""
import sys
import os
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.face_models import load_mtcnn


def register_face(name, num_images=10):
    """
    Register a new face by capturing images from webcam.
    
    Args:
        name: Person's name
        num_images: Number of images to capture
    """
    # Create person's directory
    person_dir = os.path.join("dataset", name)
    os.makedirs(person_dir, exist_ok=True)
    
    print(f"\n📸 Registering: {name}")
    print(f"   Capturing {num_images} images...")
    print(f"   Saving to: {person_dir}")
    
    # Load MTCNN for face detection
    print("\nLoading face detection model...")
    mtcnn = load_mtcnn(device="cpu")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return False
    
    print("\n" + "=" * 60)
    print("Press SPACE to capture image, ESC to cancel")
    print("=" * 60)
    
    captured = 0
    
    while captured < num_images:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error reading from webcam")
            break
        
        # Display frame
        display_frame = frame.copy()
        
        # Try to detect face
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_tensor = mtcnn(rgb)
        
        # Draw status
        status_text = f"Captured: {captured}/{num_images}"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if face_tensor is not None:
            cv2.putText(display_frame, "Face Detected!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(display_frame, (10, 90), (630, 470), (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No face detected", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Register Face - Press SPACE to capture", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n❌ Registration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        elif key == 32:  # SPACE
            if face_tensor is not None:
                # Save image
                img_path = os.path.join(person_dir, f"{name}_{captured+1}.jpg")
                cv2.imwrite(img_path, frame)
                captured += 1
                print(f"✅ Captured image {captured}/{num_images}")
            else:
                print("⚠️  No face detected, try again")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully registered {name}!")
    print(f"   {captured} images saved to {person_dir}")
    print("\n📝 Next steps:")
    print("   1. Run: python scripts/vectorize_faceset.py")
    print("   2. Run: python scripts/build_faiss_index.py")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Face Registration System")
    print("=" * 60)
    
    name = input("\nEnter person's name: ").strip()
    
    if not name:
        print("❌ Name cannot be empty")
        sys.exit(1)
    
    # Check if person already exists
    person_dir = os.path.join("dataset", name)
    if os.path.exists(person_dir):
        response = input(f"⚠️  {name} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("❌ Registration cancelled")
            sys.exit(0)
    
    # Get number of images
    try:
        num_images = int(input("Number of images to capture (default 10): ") or "10")
        if num_images < 1:
            num_images = 10
    except ValueError:
        num_images = 10
    
    # Register face
    success = register_face(name, num_images)
    
    if not success:
        sys.exit(1)
