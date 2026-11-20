"""Create the five fictitious demo patients in demo_patients/.

Each demo scan is a real MRI slice from the public test set (Kaggle Brain Tumor MRI Dataset), analysed by the
trained models exactly as an upload would be, so the stored predictions, heatmaps and EAA-IoU values are real
model outputs. Names, dates and clinical notes are invented for the demonstration.

Usage:  MRI_DATA=/path/to/MRI_DATASET MODELS_DIR=/path/to/models python seed_demo_patients.py
"""

import json
import os
import shutil

from PIL import Image

os.environ.setdefault(
    "PATIENT_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_patients")
)
import patients  # noqa: E402  (PATIENT_DATA_DIR must be set first)
from analysis import analyse  # noqa: E402

DATA = os.environ.get("MRI_DATA", "MRI_DATASET")
OUT = patients.DATA_DIR
RADIOLOGIST = "Dr. Sanjeewa Gunawardena (Consultant Radiologist)"


def test_image(cls, name):
    return Image.open(os.path.join(DATA, "Testing", cls, name)).convert("RGB")


def reviewed(final, date, note):
    return {
        "status": "confirmed",
        "radiologist": RADIOLOGIST,
        "final_diagnosis": final,
        "note": note,
        "date": date,
    }


PENDING = {"status": "pending", "radiologist": None, "final_diagnosis": None, "note": None, "date": None}

PATIENTS = [
    dict(
        id="P-0001",
        name="Nimal Perera",
        name_si="නිමල් පෙරේරා",
        sex="Male",
        dob="1972-03-14",
        phone="077 451 2286",
        city="Colombo 05",
        blood_group="O+",
        allergies="None known",
        referring_clinician="Dr. Ruwan Fernando (Neurologist)",
        history="Progressive headaches for three months and one generalised seizure in May 2026. "
        "No previous neurological illness. Started on levetiracetam.",
        scans=[
            (
                "2026-06-12",
                "MRI brain, contrast-enhanced",
                "glioma",
                "Te-gl_246.jpg",
                "Initial scan after seizure.",
                reviewed(
                    "Glioma",
                    "2026-06-13",
                    "Enhancing intra-axial lesion consistent with glioma. Referred to neurosurgery.",
                ),
            ),
            (
                "2026-09-03",
                "MRI brain, follow-up",
                "glioma",
                "Te-gl_9.jpg",
                "Follow-up after biopsy.",
                reviewed("Glioma", "2026-09-04", "Lesion persists. Continue oncology follow-up."),
            ),
        ],
    ),
    dict(
        id="P-0002",
        name="Kumari Jayasinghe",
        name_si="කුමාරි ජයසිංහ",
        sex="Female",
        dob="1980-07-22",
        phone="071 338 9047",
        city="Kandy",
        blood_group="A+",
        allergies="Penicillin",
        referring_clinician="Dr. Nadeesha Herath (General Physician)",
        history="Frontal headaches and occasional blurred vision. Hypertension, well controlled.",
        scans=[
            (
                "2026-05-20",
                "MRI brain, contrast-enhanced",
                "meningioma",
                "Te-me_58.jpg",
                "Referred for persistent headaches.",
                reviewed("Meningioma", "2026-05-21", "Well-defined extra-axial mass, typical of meningioma."),
            ),
            (
                "2026-10-08",
                "MRI brain, follow-up",
                "meningioma",
                "Te-me_294.jpg",
                "Six-month surveillance scan.",
                reviewed("Meningioma", "2026-10-09", "No significant change in size. Continue surveillance."),
            ),
        ],
    ),
    dict(
        id="P-0003",
        name="Sunil Bandara",
        name_si="සුනිල් බණ්ඩාර",
        sex="Male",
        dob="1965-11-02",
        phone="076 902 1153",
        city="Galle",
        blood_group="B+",
        allergies="None known",
        referring_clinician="Dr. Chathura Silva (Endocrinologist)",
        history="Reduced peripheral vision and raised prolactin on blood tests. Type 2 diabetes.",
        scans=[
            (
                "2026-07-15",
                "MRI pituitary, contrast-enhanced",
                "pituitary",
                "Te-pi_74.jpg",
                "Investigation of visual field loss.",
                reviewed("Pituitary tumour", "2026-07-16", "Sellar mass consistent with pituitary adenoma."),
            ),
            (
                "2026-10-30",
                "MRI pituitary, follow-up",
                "pituitary",
                "Te-pi_111.jpg",
                "Review after three months of medical treatment.",
                reviewed("Pituitary tumour", "2026-10-31", "Mass still present. Endocrine review arranged."),
            ),
        ],
    ),
    dict(
        id="P-0004",
        name="Dilani Wickramasinghe",
        name_si="දිලානි වික්‍රමසිංහ",
        sex="Female",
        dob="1993-01-30",
        phone="070 614 7720",
        city="Kurunegala",
        blood_group="AB+",
        allergies="None known",
        referring_clinician="Dr. Ruwan Fernando (Neurologist)",
        history="Recurrent migraine with aura. Normal neurological examination. Scan requested to exclude a structural cause.",
        scans=[
            (
                "2026-08-22",
                "MRI brain, non-contrast",
                "notumor",
                "Te-no_65.jpg",
                "Exclude structural cause of migraine.",
                reviewed("No tumour", "2026-08-23", "Normal study. No mass lesion."),
            ),
        ],
    ),
    dict(
        id="P-0005",
        name="Chaminda Rathnayake",
        name_si="චමින්ද රත්නායක",
        sex="Male",
        dob="1968-05-09",
        phone="072 287 5531",
        city="Matara",
        blood_group="O-",
        allergies="Sulfonamides",
        referring_clinician="Dr. Nadeesha Herath (General Physician)",
        history="New weakness of the right arm and word-finding difficulty over four weeks. Previous scan in March 2026 was normal.",
        scans=[
            (
                "2026-03-10",
                "MRI brain, non-contrast",
                "notumor",
                "Te-no_330.jpg",
                "Scan for dizziness.",
                reviewed("No tumour", "2026-03-11", "Normal study."),
            ),
            (
                "2026-11-18",
                "MRI brain, contrast-enhanced",
                "glioma",
                "Te-gl_1.jpg",
                "Urgent scan for new focal weakness.",
                PENDING,
            ),
        ],
    ),
]

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "scans"))
records = []
for p in PATIENTS:
    scans = []
    for k, (date, study, cls, fn, notes, review) in enumerate(p.pop("scans"), 1):
        response, original, maps, consensus = analyse(test_image(cls, fn))
        scan_id = f"{p['id']}-S{k}"
        orig_url, heatmaps, cons_url = patients.save_scan_images(scan_id, original, maps, consensus)
        response["xai"].update(original=orig_url, heatmaps=heatmaps, consensus=cons_url)
        scans.append(
            dict(
                id=scan_id,
                date=date,
                study=study,
                notes=notes,
                source_image=f"Testing/{cls}/{fn}",
                result=response,
                review=review,
            )
        )
        print(
            p["id"],
            date,
            cls,
            "->",
            response["final_class"],
            f"{response['confidence']:.3f}",
            "EAA",
            round(response["xai"]["eaa_iou"], 3),
            "alert" if response["xai"]["alert"] else "",
        )
    records.append({**p, "scans": scans})
json.dump(
    {
        "demo": True,
        "note": "Fictitious demonstration patients. MRI slices from the public Kaggle test set.",
        "patients": records,
    },
    open(os.path.join(OUT, "patients.json"), "w"),
    indent=1,
    ensure_ascii=False,
)
