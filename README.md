### Enhanced Flask CCTV Surveillance System

An end-to-end Flask web app that analyzes CCTV videos with Ultralytics YOLO, multi-person tracking, pose estimation (MediaPipe), hand detection, polygonal regions, and coin-insertion event detection. It outputs an annotated video, JSON stats, and CSV reports.

### Features
- **Object detection**: Ultralytics YOLO (persons only)
- **Multi-target tracking**: Stable ID assignment, occlusion handling, path prediction
- **Pose estimation**: Standing / Sitting / Walking (MediaPipe)
- **Hand detection**: Detects hands per person crop
- **Region analytics**: User-defined polygons, entries/exits, occupancy
- **Coin insertion detection**: Dwell-based event detection when hand stays inside a region
- **Reports**: Annotated video, results JSON, person logs CSV, coin insertions CSV

### Requirements
- Python 3.10+
- Windows, macOS, or Linux
- Optional GPU (CUDA) for faster processing

### Installation
1) Create and activate a virtual environment

Windows (PowerShell):
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS/Linux (bash):
```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If the PyTorch installation in `requirements.txt` fails on your machine, install a compatible build from the official selector, then reinstall the rest:
- CPU-only example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
- CUDA 11.8 example (Windows/Linux with NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3) Weights

This app looks for `yolo11n.pt` in the `app/` folder. A fallback to `yolov8n.pt` is implemented if `yolo11n.pt` is missing. You can replace the file with your own Ultralytics YOLO weights.

### Run
From the `app/` directory:
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

### Usage
- Upload a CCTV video on the main page.
- The app extracts the first frame to let you draw polygonal regions.
- Click Process to run detection, tracking, pose/hand analysis, and coin insertion events.
- Navigate to the results page to stream the processed video and download CSVs.

### Output
- `results/processed_<video_id>.mp4` (or `.avi` if codec fallback)
- `results/results_<video_id>.json`
- `results/person_logs_<video_id>.csv`
- `results/coin_insertions_<video_id>.csv`

The folders `uploads/` and `results/` are created automatically and are ignored by Git.

### Project structure (app/)
```
app/
  app.py                # Flask app and video analytics pipeline
  templates/
    index.html          # UI for upload/regions
    results.html        # Results view
  uploads/              # Temporary user uploads (ignored by Git)
  results/              # Outputs: video, JSON, CSV (ignored by Git)
  yolo11n.pt            # YOLO weights (optional if you prefer other weights)
  requirements.txt
  .gitignore
  LICENSE
  README.md
```

### API endpoints (selected)
- `POST /upload_temp` – upload a video and get a `video_id`
- `GET /get_first_frame/<video_id>` – first frame + sizes for region setup
- `POST /process` – process video with regions, progress callback
- `GET /progress/<video_id>` – processing progress
- `GET /get_results/<video_id>` – aggregated JSON results
- `GET /video/<video_id>` – stream processed video
- `GET /download/<filename>` – download output files

### License
MIT License (see `LICENSE`).

### Acknowledgements
- Ultralytics YOLO
- MediaPipe
- OpenCV

