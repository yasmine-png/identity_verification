import base64
import requests

# Fonction pour encoder une image en base64
def encode_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

# Ta clé API (tu l’as bien mise ici)
api_key = "05476a69-db31-4539-88b1-d2e8bbf4dcba" 

# URL de l’API de vérification
url = "http://localhost:8000/api/v1/verification/verify"

# En-têtes avec la clé API
headers = {
    "x-api-key": api_key,
    "Content-Type": "application/json"
}

# ⚠️ Remplace les chemins Windows par des chemins Linux (WSL)
payload = {
    "source_image": encode_image("/mnt/c/Users/hp/Downloads/ouma/image1.19_1.jpg"),
    "target_image": encode_image("/mnt/c/Users/hp/Downloads/ouma/image1.15.e_1.jpg")
}

# Envoi de la requête
response = requests.post(url, json=payload, headers=headers)

# Récupération du résultat en JSON
result = response.json()

# Extraction de la similarité
similarity = result['result'][0]['face_matches'][0]['similarity']

# Affichage du résultat
if similarity >= 0.6:
    print("✅ Faces match!")
else:
    print("❌ Faces do not match.")

# Affichage du résultat complet pour le débogage
print(result)
