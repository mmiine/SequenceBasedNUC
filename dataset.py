import os
from tkinter import N
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

from torchvision.transforms.functional import crop
import cv2
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


def IRContrastStretching(I, Tol=0.02):
    """
    Compute contrast stretched image. Good for hot target visualization.

    Parameters:
    - I: NumPy array, input image.
    - Tol: float, percentage of intensity values to be saturated at both ends.

    Returns:
    - Optimized image: contrast stretched image in uint8 [0-255].
    """


    def strechlim(I, Tol):
        """
        Compute lower and upper limits for contrast stretching.

        Parameters:
        - I: NumPy array, input image.
        - Tol: float, percentage of intensity values to be saturated at both ends.

        Returns:
        - lowhigh: tuple, lower and upper limits for contrast stretching.
        """
        # Flatten the image array
        flat_I = I.flatten()

        # Compute the number of pixels to saturate at both ends
        num_pixels = len(flat_I)
        num_saturate = int((Tol * num_pixels) / 2)

        # Sort the flattened array
        sorted_I = np.sort(flat_I)

        # Compute lower and upper limits
        lower_limit = sorted_I[num_saturate]
        upper_limit = sorted_I[-num_saturate - 1]

        return lower_limit, upper_limit

    lowLim, highLim = strechlim(I, Tol)
    I = (np.array(I, dtype=np.float32) - lowLim) / (highLim - lowLim)
    I[np.where(I < 0)] = 0
    I[np.where(I > 1)] = 1
    return np.array(I * 255, dtype=np.uint8)


class IRDataset(Dataset):
    def __init__(self, json_file, temperatures=(10, 40)):
        # Load data from JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.info = data['info'][0]  # Assuming the first 'info' object contains dataset config
        self.sequences = self.info['sequences']
        self.root_dir = self.info['root']
        self.seq_length = self.info['length']
        self.crop_size = self.info['crop_size']  # Not used as the crop size is defined in __getitem__
        self.BB_paths = data['BB_Paths']
        self.temperatures = temperatures

        self.A, self.B = self.calculateNUC()

    def load_data(self):
        data_paths = []
        BB_paths = []

        # Walk through the directory to find files
        for root, dirs, files in os.walk(self.root_dir):
            # Make sure to only consider folders that directly contain files
            if 'BB' not in root and files:
                if files and any(f.endswith('.tiff') for f in files):
                    files = sorted([file for file in files if file.endswith('.tiff')])
                    # Only consider it if we can form at least one sequence
                    if len(files) >= self.seq_length:
                        data_paths.append((root, files))
            elif 'BB' in root:
                if len(files) == 0:
                    dictTemp = {int(folder.split('C')[0]): folder for folder in dirs}
                    BB_paths.append({'root': root, 'dirs': dictTemp})

        return data_paths, BB_paths

    def load_bb_data(self):
        # Dictionary to hold BB data paths for specified temperatures
        bb_data = {}
        for temp in self.temperatures:
            bb_temp_path = os.path.join(self.root_dir, f"BB/{temp}C_60000_NF_RAW")
            if os.path.exists(bb_temp_path):
                bb_data[temp] = [os.path.join(bb_temp_path, f) for f in sorted(os.listdir(bb_temp_path)) if f.endswith('.tiff')]
        return bb_data

    def detect_motion(self, image1, image2, threshold_min=150, threshold_max=500):
        # Compute the absolute difference between two images
        diff = np.abs(image1.astype(float) - image2.astype(float))
        # Threshold the difference to find areas with substantial change
        motion_mask = (diff > threshold_min) & (diff < threshold_max)
        # Dilate to connect object's pixels
        motion_mask = ndi.binary_dilation(motion_mask, iterations=3)
        if np.any(motion_mask):
            # Label connected components of motion mask
            labelled_array, num_features = ndi.label(motion_mask)
            if num_features > 0:
                sizes = ndi.sum(motion_mask, labelled_array, range(num_features + 1))
                max_label = sizes.argmax()
                component = (labelled_array == max_label)
                coord = np.argwhere(component)
                center_x = int(coord[:, 1].mean())
                center_y = int(coord[:, 0].mean())
                coord = (center_x, center_y)
                return True, coord
        return False, (0,0)

    def determine_sequences(self):
        sequences = []
        seq_id = 0
        for root, file_names in self.data:
            for i in tqdm(range(len(file_names) - self.seq_length + 1)):
                motion_continues = 0
                last_center = (0, 0)
                for j in range(self.seq_length - 1):
                    image1 = self.read_image(os.path.join(root, file_names[i + j]))
                    image2 = self.read_image(os.path.join(root, file_names[i + j + 1]))
                    has_motion, center = self.detect_motion(image1, image2)
                    if not has_motion and motion_continues > 0:
                        motion_continues -= 1
                        continue
                    if has_motion and self.is_motion_shifted(center, last_center):
                        motion_continues += 2
                    last_center = center
                    if motion_continues > 5:
                        sequences.append({
                            'id': seq_id,
                            'start_path': os.path.join(root, file_names[i]),
                            'motion_center': last_center
                        })
                        seq_id += 1
                        break
        return sequences

    def is_motion_shifted(self, current_center, last_center, max_shift=20):
        # Check if the motion has shifted significantly between frames
        shift_distance = np.sqrt((current_center[0] - last_center[0]) ** 2 + (current_center[1] - last_center[1]) ** 2)
        return shift_distance < max_shift

    def log_sequences(self, file_path, sequences_info, BB_info=[]):
        info = {
            "root": self.root_dir,
            "length": self.seq_length,
            "crop_size": self.crop_size
        }
        json_data = {
            "info": info,
            "sequences": sequences_info,
            "BB_Paths": BB_info
        }
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)

    def createDataset(self, saveJson, root_dir, seq_length=10, crop_size=(100, 100), temperatures=(10, 40)):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.crop_size = crop_size
        self.temperatures = temperatures
        self.data, self.BB_paths = self.load_data()
        self.sequence_info = self.determine_sequences()
        if saveJson:
            self.log_sequences(saveJson, self.sequence_info, self.BB_info)
        return self.sequence_info

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        start_path = sequence_info['start_path']
        motion_center = sequence_info['motion_center']
        x_center, y_center = motion_center

        # Load images based on start_path and sequence length
        images = []
        base_path, start_filename = os.path.split(start_path)
        start_index = int(start_filename.split('.')[0])

        for i in range(self.seq_length):
            filename = f"{start_index + i:06d}.tiff"  # Ensures filenames increment correctly
            file_path = os.path.join(base_path, filename)
            image = torch.tensor(self.read_image(file_path).astype(np.float32)) / \
                    (2 ** 14 - 1)  # Normalize for processing
            images.append(image)

        # Crop images around the motion center, ensuring crops stay within image boundaries
        height, width = image.shape[0], image.shape[1]
        crop_height, crop_width = self.crop_size

        top = max(min(y_center - crop_height // 2, height - crop_height), 0)
        left = max(min(x_center - crop_width // 2, width - crop_width), 0)

        # Adjust if the calculated top or left plus the crop size exceeds the image dimensions
        if top + crop_height > height:
            top = max(0, height - crop_height)
        if left + crop_width > width:
            left = max(0, width - crop_width)

        cropped_images = []

        for img in images:
            # Crop the image
            cropped_image = crop(img, top, left, crop_height, crop_width)
            cropped_images.append(cropped_image)

        # Crop labels which are A and B accordingly
        cropped_A = crop(self.A, top, left, crop_height, crop_width)
        cropped_B = crop(self.B, top, left, crop_height, crop_width)

        # Stack the images to create a batch of shape [seq_length, C, H, W]
        image_tensor = torch.stack(cropped_images)

        return image_tensor, cropped_A, cropped_B, motion_center

    def calculateNUC(self):
        root = self.BB_paths[0]['root']
        folders = self.BB_paths[0]['dirs']

        # load low and high temperatures
        if len(self.temperatures) == 2:
            minFolder = os.path.join(root, folders[str(self.temperatures[0])])
            minFrame = self.readAverageFrames(minFolder, 30)
            maxFolder = os.path.join(root, folders[str(self.temperatures[1])])
            maxFrame = self.readAverageFrames(maxFolder, 30)

            A = (np.mean(maxFrame) - np.mean(minFrame)) / (maxFrame - minFrame)
            B = np.mean(minFrame) - A * minFrame
        elif len(self.temperatures) == 3:
            raise NotImplementedError("2+1P NUC is not implemented.")
        else:
            raise ValueError("Temperatures are not correct for 2P NUC: {self.temperatures}")

        self.scalerA = MinMaxScaler(feature_range=(-1, 1))
        A = self.scalerA.fit_transform(A)

        self.scalerB = MinMaxScaler(feature_range=(-1, 1))
        B = self.scalerB.fit_transform(B)

        return torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)

    @staticmethod
    def read_image(path):
        img = cv2.imread(path, -1)
        if img is None:
            # todo: Error handling
            raise FileNotFoundError(f"Given image path is not correct: {path}")
        return img

    @staticmethod
    def readAverageFrames(path, noofframes):
        for i, name in enumerate(os.listdir(path)):
            if i >= noofframes:
                break
            if name.endswith('.tiff'):
                img = IRDataset.read_image(os.path.join(path, name))
                if i == 0:
                    avg = np.array(img, dtype=np.float32)
                else:
                    avg += np.array(img, dtype=np.float32)
        return avg / noofframes



class createIRDataset(IRDataset):
    def __init__(self, saveJson, root_dir, seq_length=10, crop_size=(100, 100), temperatures=(10, 40)):
        self.createDataset(saveJson, root_dir, seq_length, crop_size, temperatures)

def create_splits(dataset, train_ratio=0.7, val_ratio=0.15):
    # Load the entire dataset
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)  # Ensure all samples are used
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Randomly split the dataset into training, validation, and test datasets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def create_dataloader(dataset, sampler, batch_size, shuffle=True, num_workers=4):
    dataloader = DataLoader(
        dataset,
        sampler = sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

