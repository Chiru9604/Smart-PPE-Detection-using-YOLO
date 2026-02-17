# Safety Vest and Helmet Detection – End-to-End Video Pipeline

Detects helmets and safety vests on people in video, classifies each person as **SAFE** (helmet + vest), **PARTIAL** (only one), or **UNSAFE** (no gear). Writes annotated video and saves evidence (snapshots + CSV) for violations.

## Setup (one time)

From the project directory, install in editable mode so you can run the commands from any directory:

```powershell
cd Safety-Vest-and-Helmet-Detection
pip install -e .
```

## Model (required)

The pipeline needs a trained YOLOv8 model with classes:

- **0:** No Vest  
- **1:** Helmet  
- **2:** Vest  

**Option A – Use the original repo’s trained model**

1. Clone: `git clone https://github.com/ADiTyaRaj8969/Safety-Vest-and-Helmet-Detection.git` (original repo).
2. Copy their trained weights into this project:
   - From: `Safety-Vest-and-Helmet-Detection/Q1/runs/detect/vest_helmet_final/weights/best.pt`
   - To: `Safety-Vest-and-Helmet-Detection/weights/best.pt` (this project’s `weights/` folder).

**Option B – Train your own**

Use the original repo’s `train.py` and `Q1/data.yaml` to train; then copy `best.pt` into this project’s `weights/` folder.

**Option C – Custom path**

Set the path via env, then run from any directory:

```powershell
$env:MODEL_PATH = "C:\path\to\best.pt"
safety-vest
```

## Add video

Put your video in `videos/` (e.g. `videos/input.mp4` or any `.mp4`). The script uses the first `.mp4` found or `videos/input.mp4` if it exists.

## Run (from any directory)

After `pip install -e .`, use these commands from anywhere (no need to `cd` into the project):

**CLI pipeline**

```powershell
safety-vest
```

- Uses the first `.mp4` in the project’s `videos/` (or `videos/input.mp4`)
- Writes **`outputs/annotated.mp4`** and evidence to **`outputs/violations/`** and **`outputs/logs.csv`**

## Box colors

| Status   | Description        | Color  |
|----------|--------------------|--------|
| SAFE     | Helmet + Vest      | Green  |
| PARTIAL  | Helmet only / Vest only | Cyan / Yellow |
| UNSAFE   | No helmet, no vest | Red    |

## Web app (upload → process → download)

From any directory:

```powershell
safety-vest-web
```

Then open **http://127.0.0.1:8000** in your browser. Upload a video (MP4, AVI, MOV, MKV); when processing finishes, download the annotated video. The site uses a dark, modern layout and polls for job status.

## Project structure

```
Safety-Vest-and-Helmet-Detection/
├── app.py            # FastAPI backend (upload, job queue, download)
├── web/
│   ├── index.html    # Frontend page
│   └── static/       # style.css, app.js
├── src/
│   ├── main.py       # Video pipeline (CLI + run_pipeline for web)
│   └── evidence.py   # Snapshots + CSV
├── videos/           # Input video(s) for CLI
├── outputs/          # CLI annotated output
├── web_uploads/      # Uploaded videos (web)
├── web_outputs/      # Annotated results (web)
├── weights/          # Put best.pt here (or set MODEL_PATH)
├── safety_vest_cli.py   # Entry point for safety-vest
├── safety_vest_web.py   # Entry point for safety-vest-web
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Optional: live preview

In `src/main.py`, set `SHOW_PREVIEW = True` to show a window while processing. Press `q` to stop.
