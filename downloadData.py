import wget
import os
import zipfile

# from transformers.convert_marian_to_pytorch import unzip
#
# print('Downloading dataset...')
#
# # The URL for the dataset zip file.
# url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
#
# # Download the file (if we haven't already)
# if not os.path.exists('./cola_public_1.1.zip'):
#     wget.download(url, './cola_public_1.1.zip')

# Unzip the dataset (if we haven't already)
# if not os.path.exists('./cola_public/'):
#     !unzip cola_public_1.1.zip //ini gak tau asal usul dari mana, unzip tidak dikenal.

with zipfile.ZipFile('cola_public_1.1.zip', 'r') as zip_ref:
    zip_ref.extractall()

