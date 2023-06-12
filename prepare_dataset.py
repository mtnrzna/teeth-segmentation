import wget
import os
import shutil
import zipfile
import random
import gdown
from PIL import Image


DATASET_DIR = '/content/dataset'
IMAGES_URL = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hxt48yk462-1.zip'
MASKS_URL = 'https://github.com/ImagingYeditepe/Segmentation-of-Teeth-in-Panoramic-X-ray-Image/raw/main/Original_Masks/Orig_Masks.zip'
TEST_IMAGES_GOOGLE_DRIVE_ID = 'https://drive.google.com/uc?id=1cwbZkjNh7M66-AX9kfRJt04AfglxOyJ8' # I downloaed the 4 image given and upoad them on my google drive

def clean_dataset_dir_if_exists():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)


def make_dataset_dir_and_subdirs():
    '''Dataset directory will be as folows:
    /content/dataset
    ├── raw
    │   ├── images
    │   └── masks
    └── prepared
        ├── train
        │   ├── images
        │   │   └── img
        │   └── masks
        │       └── img
        ├── validation
        │   ├── images
        │   │   └── img
        │   └── masks
        │       └── img
        └── test
            └── img
    '''
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'raw', 'images'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'raw', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'train', 'images', 'img'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'train', 'masks', 'img'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'validation', 'images'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'validation', 'images', 'img'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'validation', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'validation', 'masks', 'img'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'test'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'prepared', 'test', 'img'), exist_ok=True)
    print("Directory /content/dataset and all its subdirectories has been created")


def download_images():
    '''This downloads the original panaromic dental images without mask'''
    destination_dir = os.path.join(DATASET_DIR, 'raw')
    wget.download(IMAGES_URL, out=destination_dir)
    print("Original images has been downloaded")


def download_masks():
    '''This downloads the masks for the images'''
    destination_dir = os.path.join(DATASET_DIR, 'raw')
    wget.download(MASKS_URL, out=destination_dir)
    print("Masks has been downloaded")


def donwload_test_images():
    '''This downloads the test images that I were given in the project to test on'''
    destination_dir = os.path.join(DATASET_DIR, 'raw', 'test-data.zip')
    gdown.download(TEST_IMAGES_GOOGLE_DRIVE_ID, destination_dir, quiet=True)
    print("Test images has been downloaded")


def extract_images():
    raw_dataset_dir = os.path.join(DATASET_DIR, 'raw')

    with zipfile.ZipFile(os.path.join(raw_dataset_dir, 'hxt48yk462-1.zip'), 'r') as zip_ref:
        zip_ref.extractall(raw_dataset_dir)

    with zipfile.ZipFile(os.path.join(raw_dataset_dir, 'DentalPanoramicXrays.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(raw_dataset_dir, 'DentalPanoramicXrays'))

    source_images_dir = os.path.join(raw_dataset_dir, 'DentalPanoramicXrays', 'Images')

    # Move the image files
    for filename in os.listdir(source_images_dir):
        source_path = os.path.join(source_images_dir, filename)
        destination_path = os.path.join(raw_dataset_dir, 'images', filename)
        shutil.move(source_path, destination_path)

    #shutil.rmtree(os.path.join(raw_dataset_dir, 'DentalPanoramicXrays'))
    #os.remove(os.path.join(raw_dataset_dir, 'DentalPanoramicXrays.zip'))

    print("Original images has been extracted")


def extract_masks():
    raw_dataset_dir = os.path.join(DATASET_DIR, 'raw')
    with zipfile.ZipFile(os.path.join(raw_dataset_dir, 'Orig_Masks.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(raw_dataset_dir, 'masks'))
    
    print("Masks has been extracted")


def extract_test_images():
    raw_dataset_dir = os.path.join(DATASET_DIR, 'raw')
    test_dir = os.path.join(DATASET_DIR, 'raw', 'test')
    with zipfile.ZipFile(os.path.join(raw_dataset_dir, 'test-data.zip'), 'r') as zip_ref:
        zip_ref.extractall(test_dir)
    
    print("Test images has been extracted")


def split_images_and_masks_into_train_and_validation(train_split=0.9):
    images_dir = os.path.join(DATASET_DIR, 'raw', 'images')
    masks_dir = os.path.join(DATASET_DIR, 'raw', 'masks')

    train_dir = os.path.join(DATASET_DIR, 'prepared', 'train')
    val_dir = os.path.join(DATASET_DIR, 'prepared', 'validation')

    image_files = os.listdir(images_dir)
    random.shuffle(image_files)
    split_index = int(train_split * len(image_files))

    # Move images and masks to train directory
    for filename in image_files[:split_index]:
        source_image = os.path.join(images_dir, filename)
        source_mask = os.path.join(masks_dir, filename)
        destination_image = os.path.join(train_dir, 'images', 'img', filename)
        destination_mask = os.path.join(train_dir, 'masks', 'img', filename)
        shutil.copy(source_image, destination_image)
        shutil.copy(source_mask, destination_mask)

    # Move images and masks to validation directory
    for filename in image_files[split_index:]:
        source_image = os.path.join(images_dir, filename)
        source_mask = os.path.join(masks_dir, filename)
        destination_image = os.path.join(val_dir, 'images', 'img', filename)
        destination_mask = os.path.join(val_dir, 'masks', 'img', filename)
        shutil.copy(source_image, destination_image)
        shutil.copy(source_mask, destination_mask)

    print("All images and masks has been splitted in train and validation")


def convert_bmp_to_png_for_test_images():
    source_test = os.path.join(DATASET_DIR, 'raw', 'test')
    destination_test = os.path.join(DATASET_DIR, 'prepared', 'test', 'img')

    for filename in os.listdir(source_test):
        if filename.endswith('.bmp'):
            bmp_path = os.path.join(source_test, filename)
            png_path = os.path.join(destination_test, os.path.splitext(filename)[0] + '.png')

            # Open the BMP image
            image = Image.open(bmp_path)

            # Convert and save as PNG
            image.save(png_path, 'PNG')

    print('Test images conversion to png completed')


def print_train_val_test_numbers():
    prepared_dataset_dir = os.path.join(DATASET_DIR, 'prepared')
    train_images_count = len(os.listdir(os.path.join(prepared_dataset_dir, 'train', 'images', 'img')))
    train_masks_count = len(os.listdir(os.path.join(prepared_dataset_dir, 'train', 'masks', 'img')))
    val_images_count = len(os.listdir(os.path.join(prepared_dataset_dir, 'validation', 'images', 'img')))
    val_masks_count = len(os.listdir(os.path.join(prepared_dataset_dir, 'validation', 'masks', 'img')))
    test_images_count = len(os.listdir(os.path.join(prepared_dataset_dir, 'test', 'img')))

    print('Our dataset contains:')
    print(f'train images: {train_images_count}, train masks: {train_masks_count}')
    print(f'validation images: {val_images_count}, validation masks: {val_masks_count}')
    print(f'test images: {test_images_count}')

def prepare():
    clean_dataset_dir_if_exists()
    make_dataset_dir_and_subdirs()
    download_images()
    download_masks()
    donwload_test_images()
    extract_images()
    extract_masks()
    extract_test_images()
    split_images_and_masks_into_train_and_validation()
    convert_bmp_to_png_for_test_images()
    print_train_val_test_numbers()