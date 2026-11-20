"""One full NeuroScan analysis: six-predictor prediction, Grad-CAM, consensus map and EAA-IoU.

Used by the /api/predict endpoint, by the patient scan upload and by seed_demo_patients.py.
"""

from fastapi import HTTPException

from model_loader import HF_REPO_ID, get_models
from xai_engine import CLASS_NAMES, EAA_HIGH, EAA_LOW, explain, predict_all


def analyse(image):
    """Full analysis of one image. Returns the API response (without images) and the images themselves."""
    models, extractors, missing = get_models()
    if not any(v is not None for v in models.values()):
        raise HTTPException(503, f"No models are loaded yet. Check that {HF_REPO_ID} has a models/ folder.")

    final, confidence, per_model, fused = predict_all(image, models, extractors)
    xai = explain(image, models, final, per_model)
    level = "high" if confidence >= 0.70 else "moderate" if confidence >= 0.50 else "low"
    response = {
        "final_class": CLASS_NAMES[final],
        "final_class_idx": final,
        "confidence": confidence,
        "confidence_level": level,
        "class_names": CLASS_NAMES,
        "class_probabilities": [float(p) for p in fused],
        "per_model": [
            {
                "name": name,
                "family": fam,
                "predicted_class": CLASS_NAMES[int(p.argmax())],
                "confidence": float(p.max()),
                "probabilities": [float(v) for v in p],
            }
            for name, fam, p in per_model
        ],
        "xai": {
            "pairwise_iou": xai["pairwise_iou"],
            "eaa_iou": xai["eaa_iou"],
            "agreement_level": xai["agreement_level"],
            "alert": xai["alert"],
            "thresholds": {"high": EAA_HIGH, "low": EAA_LOW},
        },
        "models_used": len(per_model),
        "models_missing": missing,
    }
    return response, image.resize((256, 256)), xai["maps"], xai["consensus"]
