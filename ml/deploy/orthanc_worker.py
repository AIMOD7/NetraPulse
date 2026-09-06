"""
ml/deploy/orthanc_worker.py

Polls Orthanc DICOM PACS for new studies, sends each image to the local
FastAPI /predict endpoint, and posts a FHIR DiagnosticReport to HAPI-FHIR.

Usage:
    python ml/deploy/orthanc_worker.py [--interval 30]

Prerequisites:
    docker compose up -d   (starts Orthanc + HAPI-FHIR)
    uvicorn deploy.app:app --port 8000   (FastAPI must be running)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orthanc_worker")

# ---------------------------------------------------------------------------
# Config (override via env vars for production)
# ---------------------------------------------------------------------------
ORTHANC_URL = "http://localhost:8042"
FHIR_URL = "http://localhost:8080/fhir"
NETRAPULSE_URL = "http://localhost:8000"
ORTHANC_AUTH = ("orthanc", "orthanc")   # default Orthanc credentials

DR_GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


# ---------------------------------------------------------------------------
# Orthanc helpers
# ---------------------------------------------------------------------------

def get_all_study_ids() -> list[str]:
    resp = requests.get(f"{ORTHANC_URL}/studies", auth=ORTHANC_AUTH, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_first_instance_id(study_id: str) -> str | None:
    resp = requests.get(f"{ORTHANC_URL}/studies/{study_id}", auth=ORTHANC_AUTH, timeout=10)
    resp.raise_for_status()
    study = resp.json()
    series_ids = study.get("Series", [])
    if not series_ids:
        return None
    series = requests.get(
        f"{ORTHANC_URL}/series/{series_ids[0]}", auth=ORTHANC_AUTH, timeout=10
    ).json()
    instances = series.get("Instances", [])
    return instances[0] if instances else None


def download_instance_as_image_bytes(instance_id: str) -> bytes:
    """Download instance preview as PNG bytes from Orthanc."""
    resp = requests.get(
        f"{ORTHANC_URL}/instances/{instance_id}/preview",
        auth=ORTHANC_AUTH,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# NetraPulse /predict
# ---------------------------------------------------------------------------

def classify_image(image_bytes: bytes, filename: str = "fundus.png") -> dict[str, Any]:
    resp = requests.post(
        f"{NETRAPULSE_URL}/predict",
        files={"file": (filename, io.BytesIO(image_bytes), "image/png")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# FHIR DiagnosticReport
# ---------------------------------------------------------------------------

def post_diagnostic_report(
    study_id: str,
    grade: int,
    label: str,
    confidence: list[float],
) -> dict[str, Any]:
    """Create a FHIR DiagnosticReport resource in HAPI-FHIR."""
    report = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "415068001",
                    "display": "Diabetic retinopathy screening",
                }
            ],
            "text": "NetraPulse DR Screening",
        },
        "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
        "conclusion": f"DR Grade {grade}: {label}",
        "extension": [
            {
                "url": "http://netrapulse.local/fhir/StructureDefinition/orthanc-study-id",
                "valueString": study_id,
            },
            {
                "url": "http://netrapulse.local/fhir/StructureDefinition/confidence-scores",
                "valueString": json.dumps(
                    {DR_GRADE_LABELS[i]: round(c, 4) for i, c in enumerate(confidence)}
                ),
            },
        ],
    }

    resp = requests.post(
        f"{FHIR_URL}/DiagnosticReport",
        json=report,
        headers={"Content-Type": "application/fhir+json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def poll_once(processed_ids: set[str]) -> None:
    """Process all new studies not yet seen."""
    try:
        study_ids = get_all_study_ids()
    except Exception as e:
        logger.error(f"Could not reach Orthanc at {ORTHANC_URL}: {e}")
        return

    new_ids = [sid for sid in study_ids if sid not in processed_ids]
    if not new_ids:
        logger.info("No new studies.")
        return

    logger.info(f"Found {len(new_ids)} new study(ies): {new_ids}")

    for study_id in new_ids:
        try:
            logger.info(f"[{study_id}] Fetching first instance ...")
            instance_id = get_first_instance_id(study_id)
            if not instance_id:
                logger.warning(f"[{study_id}] No instances found, skipping.")
                processed_ids.add(study_id)
                continue

            logger.info(f"[{study_id}] Downloading image (instance {instance_id}) ...")
            image_bytes = download_instance_as_image_bytes(instance_id)

            logger.info(f"[{study_id}] Sending to NetraPulse /predict ...")
            result = classify_image(image_bytes, filename=f"{instance_id}.png")
            grade = result["grade"]
            label = result["label"]
            confidence = result["confidence"]
            logger.info(f"[{study_id}] Predicted: Grade {grade} — {label}")

            logger.info(f"[{study_id}] Posting DiagnosticReport to HAPI-FHIR ...")
            fhir_resp = post_diagnostic_report(study_id, grade, label, confidence)
            fhir_id = fhir_resp.get("id", "?")
            logger.info(f"[{study_id}] ✅ DiagnosticReport created: {FHIR_URL}/DiagnosticReport/{fhir_id}")

        except Exception as e:
            logger.error(f"[{study_id}] Failed: {e}")
        finally:
            processed_ids.add(study_id)


def main(interval: int = 30) -> None:
    logger.info("NetraPulse Orthanc Worker starting ...")
    logger.info(f"  Orthanc:    {ORTHANC_URL}")
    logger.info(f"  HAPI-FHIR:  {FHIR_URL}")
    logger.info(f"  NetraPulse: {NETRAPULSE_URL}")
    logger.info(f"  Poll interval: {interval}s")
    logger.info("Press Ctrl+C to stop.\n")

    processed_ids: set[str] = set()

    while True:
        logger.info("--- Polling Orthanc ---")
        poll_once(processed_ids)
        logger.info(f"Sleeping {interval}s ...\n")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetraPulse Orthanc → FHIR worker")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()
    main(interval=args.interval)
