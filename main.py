"""Project launcher menu."""
from __future__ import annotations

# Load environment variables from .env (project root)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass


import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"


MENU = {
    "1": ("Register face images (webcam capture)", SCRIPTS / "register_face.py"),
    "2": ("Vectorize dataset -> data/face_db.npy", SCRIPTS / "vectorize_faceset.py"),
    "3": ("Build FAISS index -> faiss_index/", SCRIPTS / "build_faiss_index.py"),
    "4": ("Check DB/Index status", SCRIPTS / "check_vectors.py"),
    "5": ("Run realtime attendance (terminal)", SCRIPTS / "realtime_attendance.py"),
    "6": ("Run Streamlit app", None),
    "7": ("Exit", None),
}


def run_script(script_path: Path) -> int:
    return subprocess.call([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))


def run_streamlit() -> int:
    app = PROJECT_ROOT / "streamlit_app" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)], cwd=str(PROJECT_ROOT))


def main() -> None:
    while True:
        print("\n==============================")
        print(" Face Attendance System - Menu ")
        print("==============================")
        for key, (desc, _) in MENU.items():
            print(f"{key}. {desc}")

        choice = input("\nSelect option: ").strip()

        if choice == "7":
            print("Bye!")
            break

        if choice == "6":
            run_streamlit()
            continue

        item = MENU.get(choice)
        if not item:
            print("Invalid option.")
            continue

        _, script = item
        if script is None:
            print("Not implemented.")
            continue

        if not script.exists():
            print(f"Script not found: {script}")
            continue

        run_script(script)


if __name__ == "__main__":
    main()
