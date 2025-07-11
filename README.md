Here’s your complete `README.md` content formatted and ready to copy-paste into your project directory:

---

```markdown
# 🏟️ Cross-Camera Player Mapping System

This project maps players between a **Tacticam (top-view)** and **Broadcast (side-view)** camera using:
- 🎯 YOLOv8 object detection (`best.pt`)
- 📍 DeepSORT tracking
- 👕 OSNet-based Re-Identification
- 🔄 Automatic Homography Estimation
- 📊 Evaluation + Annotated Video Output

---

## 📁 Project Structure

```

.
├── INPUT/                       # Put your videos here
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── OUTPUT/                      # All output files will be saved here
│   ├── final\_mapping.json
│   ├── broadcast\_annotated.mp4
│   ├── tacticam\_annotated.mp4
│   └── mapping\_evaluation\_report.json
├── best.pt                      # YOLOv8 custom weights
├── main.py                      # Full pipeline execution
├── visualize\_mapping.py         # Visualization + evaluation
└── requirements.txt             # Python dependencies

````

---

## ✅ Setup Instructions

### 1. 🐍 Create a virtual environment

```bash
python -m venv venv
# Activate it:
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
````

### 2. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt` should include:

```
ultralytics
torch
torchvision
torchreid
opencv-python
deep_sort_realtime
scipy
numpy
tqdm
gdown
matplotlib
```

> 🔁 Pretrained OSNet weights auto-downloaded via `torchreid`.

---

## 🚀 How to Run

### 🔧 Step 1: Run full pipeline

```bash
python main.py
```

* Performs detection, tracking, homography estimation, and Re-ID
* Saves mapping in `OUTPUT/final_mapping.json`

---

### 🎥 Step 2: Annotate videos & evaluate

```bash
python visualize_mapping.py
```

* Generates:

  * `broadcast_annotated.mp4`
  * `tacticam_annotated.mp4`
  * `mapping_evaluation_report.json`
  * `mapping_eval_chart.png` (optional)

> 🟩 Good matches are green, mismatches are red, with match confidence overlayed.

---

## 📊 Evaluation Example

```json
{
  "2": {
    "mapped_broadcast_id": "2",
    "euclidean_distance": 13.85,
    "confidence": 0.88
  },
  ...
  "_summary": {
    "mean_distance": 41.3,
    "std_distance": 20.5,
    "max_distance": 93.7,
    "worst_match": "12"
  }
}
```

---

## ⚠️ Notes

* Input videos must be named:

  * `INPUT/broadcast.mp4`
  * `INPUT/tacticam.mp4`
* Ensure your YOLO model (`best.pt`) is trained to detect players (and optionally ball)
* Works on CPU or GPU (CUDA auto-detected)

---

## 🔧 Optional Improvements

* Iterative homography refinement via RANSAC (WIP)
* Match heatmap or interactive UI with Streamlit
* Add ground-truth comparison metrics (accuracy, recall, etc.)

---

## 🆘 Need Help?

* Use Python 3.8 to 3.11
* Run inside a clean virtual environment
* Check YOLO model format and video paths

---

```

Let me know if you'd like this written to a `README.md` file or with GitHub badges, links, or example images.
```
