import pymongo

# Connectez-vous à MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["identity_verification"]  # Updated to match Django database name
collection = db["identityinfo"]  # Updated to match Django model collection name

# Affichez les données
for doc in collection.find():
    print(doc)
