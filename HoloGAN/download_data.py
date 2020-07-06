import os
from zipfile import ZipFile

def download_dataset():
    os.system("pip install gdown")
    os.system("gdown https://drive.google.com/uc?id=1JKAluJEagidnUYin77yjoiN_FW63zuZj")

def extract_dataset(path_to_zip):
    print("Extracting...")
    path_to_extract = "../dataset/celebA/"
    with ZipFile(path_to_zip, "r") as zipObj:
        zipObj.extractall(path_to_extract)
    print("Completed.")

if __name__ == "__main__":
    path_to_zip = "img_align_celeba.zip"
    if not os.path.exists(path_to_zip):
        download_dataset()
    extract_dataset(path_to_zip)
