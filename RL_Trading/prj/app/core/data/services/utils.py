import gzip
import os
import shutil
import zipfile


def extract_zip(zip_root: str, zip_file_name: str) -> str:
    zip_file_path = os.path.join(zip_root, zip_file_name)
    csv_file_path = zip_file_path.replace('zip', 'csv')
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_root)
    return csv_file_path


def extract_gzip(gzip_root: str, gzip_file_name: str) -> str:
    gzip_file_path = os.path.join(gzip_root, gzip_file_name)
    csv_file_path = gzip_file_path.replace('.gz', '')
    with gzip.open(gzip_file_path, 'rb') as f_in:
        with open(csv_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return csv_file_path
