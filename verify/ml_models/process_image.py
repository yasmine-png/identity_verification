import logging
import os
import glob
from pathlib import Path
import shutil
import subprocess
from paddleocr import PaddleOCR
from pymongo import MongoClient
from verify.models import IdentityInfo

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définir les chemins requis
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIXED_OUTPUT_PATH = BASE_DIR / "verify" / "ml_models" / "yolo_training" / "exp_fixed"
CROPS_DIR = FIXED_OUTPUT_PATH / "crops"

# Configuration OCR
ocr_config = {
    'ID': {'lang': 'fr', 'use_angle_cls': False},
    'Nom': {'lang': 'ar', 'use_angle_cls': True},
    'Prénom': {'lang': 'ar', 'use_angle_cls': True},
}

def correct_arabic(text):
    """Corrige l'inversion gauche-droite du texte arabe."""
    if not isinstance(text, str):
        return text
    arabic_chars = set(chr(i) for i in range(0x0600, 0x06FF + 1))
    if any(c in arabic_chars for c in text):
        return text[::-1]
    return text

def process_image_with_yolo(image_path):
    """
    Exécute YOLOv5 pour détecter les zones d’intérêt et sauvegarder les crops.
    :param image_path: Chemin de l’image à traiter.
    :return: Chemin du dossier contenant les crops générés.
    """
    try:
        YOLO_TRAINING_DIR = BASE_DIR / "verify" / "ml_models" / "yolo_training"
        YOLO_SCRIPT = YOLO_TRAINING_DIR / "yolov5" / "detect.py"
        YOLO_WEIGHTS = YOLO_TRAINING_DIR / "yolov5" / "runs" / "train" / "exp29" / "weights" / "best.pt"

        if FIXED_OUTPUT_PATH.exists():
            shutil.rmtree(FIXED_OUTPUT_PATH)
            logger.info(f"Ancien dossier supprimé: {FIXED_OUTPUT_PATH}")

        FIXED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dossier créé: {FIXED_OUTPUT_PATH}")

        for class_name in ['ID', 'Nom', 'Prénom']:
            (CROPS_DIR / class_name).mkdir(parents=True, exist_ok=True)

        command = [
            "python", str(YOLO_SCRIPT),
            "--weights", str(YOLO_WEIGHTS),
            "--source", str(image_path),
            "--save-crop",
            "--save-txt",
            "--conf", "0.4",
            "--project", str(YOLO_TRAINING_DIR),
            "--name", "exp_fixed",
            "--exist-ok"
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        logger.info(f"YOLO stdout: {result.stdout}")
        logger.info(f"YOLO stderr: {result.stderr}")

        if result.returncode != 0:
            raise RuntimeError(f"YOLO process failed.\nStderr: {result.stderr}")

        if not CROPS_DIR.exists() or not any(CROPS_DIR.iterdir()):
            raise RuntimeError(f"YOLO n’a pas généré de crops valides dans {CROPS_DIR}")

        return str(CROPS_DIR)

    except Exception as e:
        if FIXED_OUTPUT_PATH.exists():
            shutil.rmtree(FIXED_OUTPUT_PATH)
        logger.error(f"Erreur dans YOLO: {str(e)}")
        raise RuntimeError(f"Erreur dans YOLO: {str(e)}")

def extract_text_from_crops(crops_dir):
    """
    Utilise PaddleOCR pour extraire les textes des images recadrées.
    :param crops_dir: Chemin du dossier contenant les crops.
    :return: Liste de dictionnaires contenant les champs extraits.
    """
    aggregated_data = {}

    ocr_models = {
        class_name: PaddleOCR(**ocr_config[class_name])
        for class_name in ['ID', 'Nom', 'Prénom']
    }

    for class_name in ['ID', 'Nom', 'Prénom']:
        class_path = os.path.join(crops_dir, class_name)
        logger.info(f"Traitement des images dans {class_path}")
        if not os.path.exists(class_path):
            continue

        for crop_path in glob.glob(f"{class_path}/*.jpg"):
            try:
                ocr = ocr_models[class_name]
                result = ocr.ocr(crop_path)
                logger.info(f"Résultat brut pour {crop_path}: {result}")

                if result and result[0] and result[0][0] and len(result[0][0]) > 1:
                    text = result[0][0][1][0]
                else:
                    raise ValueError("Aucun texte détecté")

                if class_name == 'ID':
                    text = ''.join(c for c in text if c.isdigit())
                    if not text:
                        raise ValueError("Aucun chiffre détecté")
                elif class_name in ['Nom', 'Prénom']:
                    text = correct_arabic(text)

                base_name = os.path.basename(crop_path)
                if base_name not in aggregated_data:
                    aggregated_data[base_name] = {}
                if class_name == 'ID':
                    aggregated_data[base_name]['id'] = text
                elif class_name == 'Nom':
                    aggregated_data[base_name]['name'] = text
                elif class_name == 'Prénom':
                    aggregated_data[base_name]['prenom'] = text

            except ValueError as ve:
                logger.warning(f"{crop_path}: {ve}")
            except Exception as e:
                logger.error(f"Erreur OCR pour {crop_path}: {e}")

    combined_results = []
    for key, value in aggregated_data.items():
        if 'id' in value and 'name' in value and 'prenom' in value:
            combined_results.append(value)

    try:
        insert_results_into_mongodb(combined_results)
        logger.info(f"Inserted {len(combined_results)} combined OCR results into MongoDB.")
    except Exception as e:
        logger.error(f"Failed to insert combined OCR results into MongoDB: {e}")

    return combined_results

def insert_results_into_mongodb(results):
    """
    Insère les résultats OCR dans MongoDB uniquement si l'ID existe dans la base Django.
    :param results: Liste de dictionnaires contenant les résultats OCR.
    """
    try:
        client = MongoClient("mongodb://admin:admin@localhost:27017/")
        db = client["identity_verification_db"]
        collection = db["identityinfo"]

        filtered_results = []
        for doc in results:
            if IdentityInfo.objects.filter(id_number=doc['id']).exists():
                filtered_results.append(doc)

        if filtered_results:
            collection.insert_many(filtered_results)
            logger.info(f"{len(filtered_results)} documents insérés dans MongoDB.")
        else:
            logger.warning("Aucun résultat à insérer.")

    except Exception as e:
        logger.error(f"Erreur de connexion ou d'insertion MongoDB: {e}")
