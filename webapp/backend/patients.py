"""Patient records for NeuroScan.

Each patient has a profile and a scan history. Every scan keeps the full analysis that NeuroScan produced at the
time (prediction, probabilities, votes, Grad-CAM images, EAA-IoU, alert) and the radiologist's review.

The five patients shipped in demo_patients/ are fictitious demonstration records; their MRI slices come from
the public test set and their analyses were produced by the trained models (seed_demo_patients.py).
At start-up the demo records are copied to PATIENT_DATA_DIR, where new patients and scans are also saved.
"""

import datetime as dt
import io
import json
import os
import shutil
import threading
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "demo_patients")
DATA_DIR = os.environ.get("PATIENT_DATA_DIR", "/tmp/neuroscan_patients")
DB = os.path.join(DATA_DIR, "patients.json")
FILES_URL = "/patient-files"
_lock = threading.Lock()

router = APIRouter(prefix="/api/patients")


def init_store():
    """Copy the demo records on first start (or after a redeploy, when the data folder is empty)."""
    if not os.path.exists(DB):
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copytree(DEMO_DIR, DATA_DIR, dirs_exist_ok=True)


def mount_files(app):
    init_store()
    app.mount(FILES_URL, StaticFiles(directory=os.path.join(DATA_DIR, "scans")), name="patient-files")


def _load():
    with open(DB) as f:
        return json.load(f)


def _save(data):
    tmp = DB + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, DB)


def _patient(data, pid):
    for p in data["patients"]:
        if p["id"] == pid:
            return p
    raise HTTPException(404, f"Patient {pid} not found")


def age(dob):
    d = dt.date.fromisoformat(dob)
    t = dt.date.today()
    return t.year - d.year - ((t.month, t.day) < (d.month, d.day))


def save_scan_images(scan_id, original, maps, consensus):
    """Store the images of one analysis and return their URLs."""
    folder = os.path.join(DATA_DIR, "scans", scan_id)
    os.makedirs(folder, exist_ok=True)

    def put(img, name):
        if img is None:
            return None
        img.save(os.path.join(folder, name + ".png"))
        return f"{FILES_URL}/{scan_id}/{name}.png"

    return (
        put(original, "original"),
        [{"name": n, "image": put(img, n.lower())} for n, img in maps.items()],
        put(consensus, "consensus"),
    )


def summary(p):
    scans = sorted(p["scans"], key=lambda s: s["date"])
    last = scans[-1] if scans else None
    return {
        "id": p["id"],
        "name": p["name"],
        "name_si": p.get("name_si"),
        "sex": p["sex"],
        "age": age(p["dob"]),
        "city": p.get("city"),
        "n_scans": len(scans),
        "last_scan_date": last["date"] if last else None,
        "last_result": last["result"]["final_class"] if last else None,
        "last_confidence": last["result"]["confidence"] if last else None,
        "last_alert": bool(last["result"]["xai"]["alert"]) if last else False,
        "pending_reviews": sum(s["review"]["status"] == "pending" for s in scans),
    }


@router.get("")
def list_patients():
    data = _load()
    return {"demo": data.get("demo", True), "patients": [summary(p) for p in data["patients"]]}


@router.get("/{pid}")
def get_patient(pid: str):
    p = _patient(_load(), pid)
    return {**p, "age": age(p["dob"]), "scans": sorted(p["scans"], key=lambda s: s["date"], reverse=True)}


class NewPatient(BaseModel):
    name: str
    name_si: str | None = None
    sex: str
    dob: str
    phone: str | None = None
    city: str | None = None
    referring_clinician: str | None = None
    history: str | None = None


@router.post("")
def create_patient(body: NewPatient):
    try:
        dt.date.fromisoformat(body.dob)
    except ValueError:
        raise HTTPException(400, "Date of birth must be YYYY-MM-DD.")
    with _lock:
        data = _load()
        n = max(int(p["id"].split("-")[1]) for p in data["patients"]) + 1 if data["patients"] else 1
        p = {"id": f"P-{n:04d}", **body.model_dump(), "blood_group": None, "allergies": None, "scans": []}
        data["patients"].append(p)
        _save(data)
    return summary(p)


def add_scan(pid, image, study, notes, analyse):
    """Run the full analysis on a new image and add it to the patient's history."""
    response, original, maps, consensus = analyse(image)
    scan_id = f"{pid}-{uuid.uuid4().hex[:8]}"
    orig_url, heatmaps, cons_url = save_scan_images(scan_id, original, maps, consensus)
    response["xai"].update(original=orig_url, heatmaps=heatmaps, consensus=cons_url)
    scan = {
        "id": scan_id,
        "date": dt.date.today().isoformat(),
        "study": study or "MRI brain, single slice",
        "notes": notes or "",
        "result": response,
        "review": {
            "status": "pending",
            "radiologist": None,
            "final_diagnosis": None,
            "note": None,
            "date": None,
        },
    }
    with _lock:
        data = _load()
        _patient(data, pid)["scans"].append(scan)
        _save(data)
    return scan


def register_scan_upload(analyse):
    """Upload endpoint; `analyse(image)` is provided by main.py so the models are loaded once."""

    @router.post("/{pid}/scans")
    async def upload_scan(
        pid: str, file: UploadFile = File(...), study: str = Form(""), notes: str = Form("")
    ):
        _patient(_load(), pid)
        if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
            raise HTTPException(400, "Please upload a JPEG or PNG image.")
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception:
            raise HTTPException(400, "The file could not be read as an image.")
        return add_scan(pid, image, study, notes, analyse)


class Review(BaseModel):
    status: str  # confirmed | corrected
    final_diagnosis: str
    radiologist: str
    note: str | None = None


@router.post("/{pid}/scans/{sid}/review")
def review_scan(pid: str, sid: str, body: Review):
    if body.status not in ("confirmed", "corrected"):
        raise HTTPException(400, "Status must be 'confirmed' or 'corrected'.")
    with _lock:
        data = _load()
        for s in _patient(data, pid)["scans"]:
            if s["id"] == sid:
                s["review"] = {**body.model_dump(), "date": dt.date.today().isoformat()}
                _save(data)
                return s
    raise HTTPException(404, f"Scan {sid} not found")
