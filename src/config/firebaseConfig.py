import firebase_admin
from firebase_admin import credentials, storage
import os

_bucket = None

def get_bucket():
    global _bucket

    if _bucket is not None:
        return _bucket

    if not firebase_admin._apps:
        cred = credentials.Certificate(
            os.getenv("FIREBASE_SERVICE_ACCOUNT")
        )

        firebase_admin.initialize_app(cred, {
            "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET")
        })

    _bucket = storage.bucket()
    return _bucket
