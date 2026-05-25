# Traffic Sign Recognition (ITS)

Moderate stack for major project: **FastAPI** backend, **YOLO** model, **SQLite** database, **HTML/CSS/JS** frontend.

## Tech stack

| Layer | Technology |
|--------|------------|
| Backend | Python, FastAPI, Uvicorn |
| ML | Ultralytics YOLO (`best.pt`) |
| Database | SQLite (`detections.db`) |
| Frontend | HTML, CSS, JavaScript |

No Docker or React required.

## Run

```bash
cd Traffic-Sign-Recognition-for-Intelligent-Transport-System
pip install -r requirements.txt
python main.py
```

Open **http://127.0.0.1:8080**

1. **Upload image**
2. **Analyze image**
3. View sign name + confidence %

### Team photos

Developer photos live in `static/img/developers/` (`uday.png`, `samar.jpeg`, `paras.png`, `Sanidhya.png`).

## Project files

- `main.py` — API + model + database
- `best.pt` — trained weights
- `templates/` + `static/` — website UI
- `labels.csv` — dataset labels (reference)
