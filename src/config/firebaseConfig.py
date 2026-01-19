import json
import firebase_admin
from firebase_admin import credentials, storage
import os

_bucket = None

def get_bucket():
    global _bucket

    if _bucket is not None:
        return _bucket

    if not firebase_admin._apps:
        if os.path.exists("src/config/firebase-key.json"):
            cred = credentials.Certificate("src/config/firebase-key.json")
        else:
            # 👉 PRODUCCIÓN (Render)
            cred = credentials.Certificate(
                json.loads(os.getenv("FIREBASE_SERVICE_ACCOUNT"))
            )

        firebase_admin.initialize_app(cred, {
            "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET")
        })

    _bucket = storage.bucket()
    return _bucket
