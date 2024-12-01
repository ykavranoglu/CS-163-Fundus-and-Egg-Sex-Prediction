# Training part

import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import logging
import numpy as np
import random
import pickle
from torcheval.metrics import BinaryAccuracy
import matplotlib.pyplot as plt


num_classes = 1  # doing this for binary classification
lr = 0.00001
n_epochs = 5
batch_size = 64
batch_size_test = 64
RANDOM_SEED = 1839
pretrained = True
model_name = 'resnetv2_50x1_bit.goog_in21k'

dataset_path = "./processed_data/full_256"
train_set_path = os.path.join(dataset_path, 'training')
validation_set_path = os.path.join(dataset_path, 'validation')
test_set_path = os.path.join(dataset_path, 'testing')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configure dataset
class CustomDataset(Dataset):
    def __init__(self, pickle_dir):
        self.pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('pickle') and not f.startswith('.')]
        self.data = []
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor(),  # doing this because it wasn't done when creating pickle files
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # values for imagenet
        ])

        for pickle_file in self.pickle_files:
            with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
                self.data.extend(pickle.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        image = data_point['image']
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(data_point['label'] == 'Female'), dtype=torch.float)
        additional_info = data_point['additional_info']
        return image, label, additional_info

# This one doesn't have data augmentation, has the same normalization as CustomDataset
class CustomDatasetTest(Dataset):
    def __init__(self, pickle_dir):
        self.pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('pickle') and not f.startswith('.')]
        self.data = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # doing this because it wasn't done when creating pickle files
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # values for imagenet

        ])

        for pickle_file in self.pickle_files:
            with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
                self.data.extend(pickle.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        image = data_point['image']
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(data_point['label'] == 'Female'), dtype=torch.float)
        additional_info = data_point['additional_info']
        return image, label, additional_info


def set_seed(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(RANDOM_SEED)

# create model and dataset, dataloader
model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes).to(device)

dataset_train = CustomDataset(train_set_path)
dataset_validation = CustomDatasetTest(validation_set_path)
dataset_test = CustomDatasetTest(test_set_path)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size_test, shuffle=False, drop_last=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=False)

# optimizer, loss_fn, learning rate scheduler
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# train and test functions
def train(dataloader_train, model, loss_fn, optimizer, device):
    model.train()

    size_of_dataset = len(dataloader_train.dataset)

    running_loss = 0.0
    for batch_id, (images, labels, additional_info) in enumerate(dataloader_train):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        predictions = model(images).sigmoid().squeeze(1)
        loss = loss_fn(predictions, labels)  # could make predictions to be probabilities as well
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_id + 1) % 5 == 0:
            images_completed = (batch_id + 1) * len(images)
            print(f'loss: {loss.item():.6f} [{images_completed:>5d}/{size_of_dataset:>5d}]')

    loss_of_epoch = running_loss / len(dataloader_train)
    print(f'Train Loss: {loss_of_epoch:.4f}')
    return loss_of_epoch


def validate(dataloader_validation, model, loss_fn, device):
    model.eval()

    running_loss = 0.0
    metric = BinaryAccuracy(threshold=0.5)
    with torch.no_grad():
        for batch_id, (images, labels, additional_info) in enumerate(dataloader_validation):
            images, labels = images.to(device), labels.to(device)

            prediction = model(images).sigmoid().squeeze(1)
            loss = loss_fn(prediction, labels)  # could make predictions to be probabilities as well

            running_loss += loss.item()

            metric.update(prediction, labels)

    loss_of_epoch = running_loss / len(dataloader_validation)
    accuracy = 100. * metric.compute().item()
    print(f'Validation Loss: {loss_of_epoch:.4f}\t Validation Accuracy: {accuracy:.2f}')
    return loss_of_epoch, accuracy

# training loop
print("Starting training...")

losses = list()
losses_validation = list()
accuracies_validation = list()
for epoch in range(n_epochs):
    print(f'\nEpoch {epoch+1}\n{"-"*28}')

    loss = train(dataloader_train, model, loss_fn, optimizer, device)
    loss_validation, accuracy_validation = validate(dataloader_validation, model, loss_fn, device)

    losses.append(loss)
    losses_validation.append(loss_validation)
    accuracies_validation.append(accuracy_validation)

    scheduler.step()

print("\n\n\nStarting testing on the test set:")
loss_test, accuracy_test = validate(dataloader_test, model, loss_fn, device)

plot_dir = './'

plt.figure(figsize=(10, 5))
plt.title("Training and validation losses")
plt.plot(losses, label="Training loss")
plt.plot(losses_validation, label="Validation loss")
plt.ylim(ymin=0)
plt.axhline(y=loss_test, color='r', linestyle='-', label="Test loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(plot_dir, f'Training_losses_256.png'))

plt.figure(figsize=(10, 5))
plt.title("Accuracies")
plt.plot(accuracies_validation, label="Validation accuracies")
plt.ylim(ymin=0)
plt.axhline(y=accuracy_test, color='r', linestyle='-', label="Test accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(plot_dir, f'Training_accuracies_256.png'))

