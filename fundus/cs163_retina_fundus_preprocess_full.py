# Processing the csv file

import os
import csv

dir_to_data = "./"


csv_name = "filtered_normal_fundus_data.csv"
csv_path = os.path.join(dir_to_data, csv_name)
data_points = list()
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    next(reader)  # to skip the header line
    for row_number, row in enumerate(reader):
        data_point_left_fundus = dict()
        data_point_right_fundus = dict()

        data_point_left_fundus['sex'] = data_point_right_fundus['sex'] = row[2]

        data_point_left_fundus['path'] = os.path.join(dir_to_data, data_point_left_fundus['sex'].lower(), row[3])
        data_point_right_fundus['path'] = os.path.join(dir_to_data, data_point_right_fundus['sex'].lower(), row[4])

        additional_info = dict()
        additional_info['id'] = row[0]
        additional_info['age'] = row[1]

        data_point_left_fundus['additional_info'] = data_point_right_fundus['additional_info'] = additional_info

        data_points.extend([data_point_left_fundus, data_point_right_fundus])

data_points = [data_point for data_point in data_points if os.path.exists(data_point['path'])]

print(len(data_points))  # Check if number of existing images are correct

# Train, validation, test split

import random
from random import shuffle
import math

random.seed(1839)

data_points_male = [data_point for data_point in data_points if data_point['sex'] == 'Male']
data_points_female = [data_point for data_point in data_points if data_point['sex'] == 'Female']

ids_set_male = set()
for data_point in data_points_male:
    id = data_point['additional_info']['id']
    ids_set_male.add(id)
ids_set_female = set()
for data_point in data_points_female:
    id = data_point['additional_info']['id']
    ids_set_female.add(id)

# Determine the size of train, validation, and test sets based off of subject ids, because every subject has exactly 2 images.
male_size = len(ids_set_male)
female_size = len(ids_set_female)
train_size_male, test_size_male = math.ceil(0.75 * male_size), math.ceil(0.125 * male_size)
validation_size_male = male_size - train_size_male - test_size_male
train_size_female, test_size_female = math.ceil(0.75 * female_size), math.ceil(0.125 * female_size)
validation_size_female = female_size - train_size_female - test_size_female

print(train_size_male / train_size_female, validation_size_male / validation_size_female, test_size_male / test_size_female)

# Sets are turned into
ids_set_male = list(ids_set_male)
ids_set_female = list(ids_set_female)
shuffle(ids_set_male)
shuffle(ids_set_female)

train_ids = ids_set_male[:train_size_male] + ids_set_female[:train_size_female]
validation_ids = ids_set_male[train_size_male+test_size_male:] + ids_set_female[train_size_female+test_size_female:]
test_ids = ids_set_male[train_size_male:train_size_male+test_size_male] + ids_set_female[train_size_female:train_size_female+test_size_female]

print(len(train_ids), len(test_ids), len(validation_ids))

data_points_train = [data_point for data_point in data_points if data_point['additional_info']['id'] in train_ids]
data_points_validation = [data_point for data_point in data_points if data_point['additional_info']['id'] in validation_ids]
data_points_test = [data_point for data_point in data_points if data_point['additional_info']['id'] in test_ids]

print(len(data_points_train), len(data_points_validation), len(data_points_test))

for data_point in data_points:
    if (data_point in data_points_train) and (data_point in data_points_validation):
        print("ERROR!!!")
    if (data_point in data_points_train) and (data_point in data_points_test):
        print("ERROR!!!")
    if (data_point in data_points_validation) and (data_point in data_points_test):
        print("ERROR!!!")



from pathlib import Path
import pickle
from PIL import Image
import torchvision.transforms as transforms
import torch
import random

img_size = 512
pickle_chunk_size = 5000

data_points = data_points_validation

def load_and_process_image(file_path, img_size):
    try:
        image = Image.open(file_path)

        shorter_edge = min(image.size)

        # transforms.ToTensor() is removed for the pickle size to be smaller
        preprocess = transforms.Compose([
            transforms.CenterCrop(shorter_edge),
            transforms.Resize((img_size, img_size))
        ])

        image = preprocess(image)
        return image

    except Exception as e:
        print(f'Error loading the image: {e}')
        return None

def chunks(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i+chunk_size]

data_points_chunks_generator = chunks(data_points, pickle_chunk_size)

output_dir = os.path.join("./processed_data")
Path(output_dir).mkdir(parents=True, exist_ok=True)

for chunk_no, chunk in enumerate(data_points_chunks_generator):
    dataset_part = list()

    for chunk_entry_no, chunk_entry in enumerate(chunk):
        if (chunk_entry_no + 1) % 50 == 0:
            print(f'Processing chunk entry no: {chunk_entry_no + 1}')

        image = load_and_process_image(chunk_entry['path'], img_size)
        data_point = {'label': chunk_entry['sex'], 'image': image, 'additional_info': chunk_entry['additional_info']}
        dataset_part.append(data_point)

    with open(os.path.join(output_dir, f'dataset_validation512_part_{chunk_no + 1}.pickle'), 'wb') as f:
        pickle.dump(dataset_part, f)
