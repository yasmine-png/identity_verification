import os
import sys
from datetime import datetime

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

from .models import IdentityInfo

# Add your project root to the path
sys.path.append('/home/oumayma/identity_verification')

# Import your ML processing functions
from verify.ml_models.process_image import process_image_with_yolo, extract_text_from_crops  # type: ignore

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED_OUTPUT_PATH = os.path.join(BASE_DIR, "ml_models/yolo_training/exp_fixed")
import subprocess
import os
import subprocess

def save_to_hadoop(local_path, hdfs_path):
    hadoop_bin = '/home/hadoop/hadoop/bin/hdfs'
    env = os.environ.copy()
    env['HADOOP_CONF_DIR'] = '/home/hadoop/hadoop/etc/hadoop'  # Set the correct config path

    try:
        mkdir_cmd = [hadoop_bin, 'dfs', '-mkdir', '-p', hdfs_path]
        put_cmd = [hadoop_bin, 'dfs', '-put', '-f', local_path, hdfs_path]

        mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, env=env)
        print(f"mkdir stdout: {mkdir_result.stdout}")
        print(f"mkdir stderr: {mkdir_result.stderr}")

        put_result = subprocess.run(put_cmd, capture_output=True, text=True, env=env)
        print(f"put stdout: {put_result.stdout}")
        print(f"put stderr: {put_result.stderr}")

        if mkdir_result.returncode != 0 or put_result.returncode != 0:
            print(f"Error saving to Hadoop: mkdir returncode {mkdir_result.returncode}, put returncode {put_result.returncode}")
        else:
            print(f"Saved {local_path} to Hadoop at {hdfs_path}")

    except subprocess.CalledProcessError as e:
        print(f"Exception saving to Hadoop: {e}")

# Upload page view
@csrf_exempt
def upload_view(request):
    if request.method == 'POST' and request.FILES.get('photos'):
        uploaded_files = request.FILES.getlist('photos')
        fs = FileSystemStorage()

        for photo in uploaded_files:
            fs.save(photo.name, photo)

        return HttpResponse('Photos uploaded successfully!')
    return render(request, 'verify/upload.html')

# Accept single photo for processing
@csrf_exempt
def accept_photo(request):
    if request.method == 'POST' and request.FILES.get('photo'):
        temp_dir = os.path.join('media', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        fs = FileSystemStorage(location=temp_dir)
        uploaded_photo = request.FILES['photo']
        temp_path = fs.save(uploaded_photo.name, uploaded_photo)
        full_temp_path = fs.path(temp_path)
        print(f"Local file exists: {os.path.exists(full_temp_path)} at {full_temp_path}")
        hdfs_target_dir = '/ids'
        print(f"Hadoop target directory: {hdfs_target_dir}")
        save_to_hadoop(full_temp_path, hdfs_target_dir)

        try:
            # Step 1: Process image with YOLO
            crops_dir = process_image_with_yolo(full_temp_path)
            print(f"Dossier crops confirmé: {os.listdir(crops_dir)}")

            # Step 2: Extract text from YOLO crops
            extracted_data = extract_text_from_crops(crops_dir)
            if not extracted_data:
                raise ValueError("Aucune donnée n'a pu être extraite de l'image")

            id_number = None
            first_name = None
            last_name = None
            photo_name = uploaded_photo.name

            # Adjusted to new combined document structure
            for item in extracted_data:
                if 'id' in item and item['id'] not in [None, '', 'Aucun texte détecté', 'ERREUR']:
                    id_number = item['id']
                if 'prenom' in item and item['prenom'] not in [None, '', 'Aucun texte détecté', 'ERREUR']:
                    first_name = item['prenom']
                if 'name' in item and item['name'] not in [None, '', 'Aucun texte détecté', 'ERREUR']:
                    last_name = item['name']

            person_exists = False
            if id_number and (first_name or last_name):
                # Check if record exists; only accept if exists
                existing = IdentityInfo.objects.filter(id_number=id_number).first()
                if existing:
                    # Person exists, proceed as usual
                    person_exists = True
                    print(f"Record with ID {id_number} exists. Accepting photo.")
                    # Save data only if person exists (do not save if not exists)
                    # To prevent race conditions, use a transaction.atomic block
                    from django.db import transaction
                    with transaction.atomic():
                        records = IdentityInfo.objects.select_for_update().filter(id_number=id_number)
                        for record in records:
                            record.photo_name = photo_name
                            record.first_name = first_name or ''
                            record.last_name = last_name or ''
                            record.save()
                else:
                    # Person does not exist, reject photo
                    person_exists = False
                    print(f"Record with ID {id_number} does not exist. Rejecting photo.")
                    return JsonResponse({
                        'status': 'error',
                        'message': 'A person with this ID does not exist in our database. Please contact us if there is any mismatch or issue.'
                    })
            else:
                print("Incomplete data, not saving to database.")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Incomplete data extracted from photo. Please try again with a clearer image.'
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Erreur lors du traitement de l'image : {e}")
            return JsonResponse({'status': 'error', 'message': str(e)})

        # Save extracted face image path in session for later verification
        request.session['extracted_face_image_path'] = os.path.join('media', 'temp', uploaded_photo.name)

        # Return extracted data in response for frontend display
        return JsonResponse({
            'status': 'success',
            'message': 'Photo processed and data saved.',
            'first_name': first_name,
            'last_name': last_name,
            'id_number': id_number,
            'person_exists': person_exists
        })

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})
