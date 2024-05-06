import itertools
import os
import pathlib
import sys
from typing import List, Tuple

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import skimage.io
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import List, Tuple, Set

from logger import create_logger


def resource_path(relative_path) -> str:
        """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
        #if getattr(sys, 'frozen', False):
        try:
            # PyInstaller создаёт временную папку _MEIPASS для ресурсов
            base_path = sys._MEIPASS
        except Exception:
            # Если приложение запущено из исходного кода, то используется обычный путь
            base_path = os.path.abspath(".")
    
        return os.path.join(base_path, relative_path)

last_person_id = 15
rev_path =  os.path.join("logs", "training", "gaze")
abs_path = pathlib.Path(resource_path(rev_path))
logger = create_logger("TrainGaze", abs_path, 'train_gaze_log.txt')



def filter_persons_by_idx(file_names: List[str], keep_person_idxs: List[int]) -> List[int]:

    keep_person_set = set(f'p{idx:02}' for idx in keep_person_idxs)  # creates a set like {'p02', 'p03', ...}
    logger.info(f"Initialized keep_person_set with IDs: {keep_person_set}")
    valid_indices = []
    unique_person_ids = set()

    for idx, file_name in enumerate(file_names):

        
        person_id = file_name.split('\\')[0]  
        unique_person_ids.add(person_id)
        #logger.info(f"Processing file {idx}: {file_name} with extracted person ID: {person_id}")

        if person_id in keep_person_set:
            valid_indices.append(idx)
            #logger.info(f"Adding index {idx} for person ID {person_id}")

    logger.info(f"Requested Person IDs Set: {keep_person_set}")
    logger.info(f"Unique Person IDs Found: {unique_person_ids}")
    logger.info(f"Filtered Indices Count: {len(valid_indices)}")
    if len(valid_indices) == 0:
        logger.info("Warning: No valid entries found. Check your dataset and filtering criteria.")

    return valid_indices, sorted(unique_person_ids)


def remove_error_data(data_path: str, file_names: List[str]) -> List[int]:
    """
    Remove erroneous data, where the gaze point is not in the screen.

    :param data_path: path to the dataset including the `not_on_screen.csv` file
    :param file_names: list of all file names
    :return: list of idxs of valid data
    """
    valid_idxs = []

    df = pd.read_csv(f'{data_path}/not_on_screen.csv')
    error_file_names = set([error_file_name[:-8] for error_file_name in df['file_name'].tolist()])
    logger.info(f"Read {len(error_file_names)} error file names from 'not_on_screen.csv'.")

    file_names = [file_name[:-4] for file_name in file_names]
    for idx, file_name in enumerate(file_names):
        #logger.info(f"Processing file {idx}: {file_name}")
        if file_name not in error_file_names:
            #logger.info(f"File {file_name} is valid. Adding index {idx}.")
            valid_idxs.append(idx)
    logger.info(f"Total valid indices collected: {len(valid_idxs)}")
    return valid_idxs


class MPIIFaceGaze(Dataset):
    """
    MPIIFaceGaze dataset with offline preprocessing (= already preprocessed)
    """

    def __init__(self, data_path: str, file_name: str, 
                 keep_person_idxs: List[int], transform=None, 
                 train: bool = False, force_flip: bool = False, 
                 use_erroneous_data: bool = False):
        if keep_person_idxs is not None:
            assert len(keep_person_idxs) > 0, "Person index list cannot be empty."
            assert max(keep_person_idxs) <= last_person_id-1, f"Person index exceeds maximum limit of {last_person_id - 1}." 
            assert min(keep_person_idxs) >= 0, "Person index cannot be less than 0."
            logger.info(f"Initialized with valid person indices: {keep_person_idxs}")
        
        self.data_path = data_path
        self.hdf5_file_name = f'{data_path}/{file_name}'
        self.h5_file = None

        self.transform = transform
        self.train = train
        self.force_flip = force_flip
        logger.info(f"Opening HDF5 file at: {self.hdf5_file_name}")
        with h5py.File(self.hdf5_file_name, 'r') as f:
            file_names = [file_name.decode('utf-8') for file_name in f['file_name_base']]
            logger.info(f"Loaded file names from HDF5: {len(file_names)} files")

        by_person_idx, unique_person_ids = filter_persons_by_idx(file_names, keep_person_idxs)
        logger.info(f"Filtered file names by person indices. Valid entries: {len(by_person_idx)}")

        if not train:
          
            self.idx2ValidIdx = by_person_idx
            logger.info(f"Non-training mode: using filtered indices directly.")
        else:
            if use_erroneous_data:
                logger.info("Training mode with erroneous data included.")
                non_error_idx = file_names
            else:
                logger.info("Removing erroneous data for training mode.")
                non_error_idx = remove_error_data(data_path, file_names)
            self.idx2ValidIdx = list(set(by_person_idx) & set(non_error_idx))
            logger.info(f"Training mode: Final valid entries count after error removal: {len(self.idx2ValidIdx)}")

        logger.info(f'Unique person IDs found: {unique_person_ids}')
        logger.info(f'Number of valid entries by person ID filter: {len(by_person_idx)}')
        logger.info(f'Number of total valid entries after processing: {len(self.idx2ValidIdx)}')
        if len(self.idx2ValidIdx) == 0:
            logger.info("Warning: No valid entries found. Check your dataset and filtering criteria.")
        

    def __len__(self) -> int:
        return len(self.idx2ValidIdx) * 2 if self.train else len(self.idx2ValidIdx)

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

    def open_hdf5(self):
        self.h5_file = h5py.File(self.hdf5_file_name, 'r')

    def __getitem__(self, idx):
        #logger.info(f"Original index received: {idx}")

        if torch.is_tensor(idx):

            idx = idx.tolist()
            #logger.info(f"Index converted from tensor to list: {idx}")

        if self.h5_file is None:
            #logger.info("Opening HDF5 file...")
            self.open_hdf5()

        augmented_person = idx >= len(self.idx2ValidIdx)
        #logger.info(f"Is augmented person? {augmented_person}")
        if augmented_person:
            original_idx = idx
            idx -= len(self.idx2ValidIdx)  # Adjust index for augmented data
            #logger.info(f"Index adjusted for augmentation from {original_idx} to {idx}")

        idx = self.idx2ValidIdx[idx]
        #logger.info(f"Valid index used for data retrieval: {idx}")

        file_name = self.h5_file['file_name_base'][idx].decode('utf-8')
        gaze_location = self.h5_file['gaze_location'][idx]
        screen_size = self.h5_file['screen_size'][idx]
        #logger.info(f"Data retrieved for file: {file_name}")
        person_idx = int(file_name.split('\\')[-3][1:])

        left_eye_image = skimage.io.imread(f"{self.data_path}/{file_name}-left_eye.png")
        left_eye_image = np.flip(left_eye_image, axis=1)
        right_eye_image = skimage.io.imread(f"{self.data_path}/{file_name}-right_eye.png")
        full_face_image = skimage.io.imread(f"{self.data_path}/{file_name}-full_face.png")
        #logger.info("Images loaded and initial flipping applied.")

        gaze_pitch = np.array(self.h5_file['gaze_pitch'][idx])
        gaze_yaw = np.array(self.h5_file['gaze_yaw'][idx])
        #logger.info("Gaze data loaded.")

        if augmented_person or self.force_flip:
            #logger.info("Applying additional flipping for augmentation...")
            person_idx += last_person_id  # fix person_idx
            left_eye_image = np.flip(left_eye_image, axis=1)
            right_eye_image = np.flip(right_eye_image, axis=1)
            full_face_image = np.flip(full_face_image, axis=1)
            gaze_yaw *= -1 # Invert yaw for flipped images

        if self.transform:
            #logger.info("Applying transformations...")
            left_eye_image = self.transform(image=left_eye_image)["image"]
            right_eye_image = self.transform(image=right_eye_image)["image"]
            full_face_image = self.transform(image=full_face_image)["image"]
        #logger.info("Data prepared for return.")
        return {    
            'file_name': file_name,
            'gaze_location': gaze_location,
            'screen_size': screen_size,

            'person_idx': person_idx,

            'left_eye_image': left_eye_image,
            'right_eye_image': right_eye_image,
            'full_face_image': full_face_image,

            'gaze_pitch': gaze_pitch,
            'gaze_yaw': gaze_yaw,
        }


def get_dataloaders(path_to_data: str, validate_on_person: int, test_on_person: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, valid and test dataset.
    The train dataset includes all persons except `validate_on_person` and `test_on_person`.

    :param path_to_data: path to dataset
    :param validate_on_person: person id to validate on during training
    :param test_on_person: person id to test on after training
    :param batch_size: batch size
    :return: train, valid and test dataset
    """
    transform = {
        'train': A.Compose([
            A.ShiftScaleRotate(p=1.0, shift_limit=0.2, scale_limit=0.1, rotate_limit=10),
            A.Normalize(),
            ToTensorV2()
        ]),
        'valid': A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
    }
    logger.info('Transforms set for train and validation.')

    train_on_persons = list(range(0, last_person_id))
    logger.info(f'Initial list of training persons: {train_on_persons}')

    if validate_on_person in train_on_persons:
        train_on_persons.remove(validate_on_person)
    if test_on_person in train_on_persons:
        train_on_persons.remove(test_on_person)
    logger.info(f'Training on persons: {train_on_persons}')
    logger.info(f'Validating on person: {validate_on_person}')
    logger.info(f'Testing on person: {test_on_person}')

    dataset_train = MPIIFaceGaze(path_to_data, 'data.h5', keep_person_idxs=train_on_persons, transform=transform['train'], train=True)
    logger.info(f'Initialized train dataset with {len(dataset_train)} entries.')
    

    dataset_valid = MPIIFaceGaze(path_to_data, 'data.h5', keep_person_idxs=[validate_on_person], transform=transform['valid'])
    logger.info(f'Initialized validation dataset with {len(dataset_valid)} entries.')
    

    dataset_test = MPIIFaceGaze(path_to_data, 'data.h5', keep_person_idxs=[test_on_person], transform=transform['valid'], use_erroneous_data=True)
    logger.info(f'Initialized test dataset with {len(dataset_test)} entries, including erroneous data.')
    
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info('DataLoaders for train, validation, and test are created.')

    return train_dataloader, valid_dataloader, test_dataloader