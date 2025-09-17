# utils.py
import os
import torch  
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    RandCropByPosNegLabeld,
    CastToTypeD,
    ToTensord,
    Compose,
    NormalizeIntensityd,
    CenterSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
)
from monai.data import Dataset, DataLoader
import numpy as np

def get_data_loader(
    data_path,
    label_path=None,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    roi_depth=96,
    roi_height=96,
    roi_width=96,
    a_min=-57.0,
    a_max=164.0,
    b_min=0.0,
    b_max=1.0,
    num_samples_per_image=1,
    transforms=None,
):
    """
    Create a MONAI DataLoader for inference.

    Args:
        data_path (str): Path to the directory containing the images.
        label_path (str, optional): Path to the directory containing the labels. Defaults to None.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.
        roi_depth (int, optional): Depth of the ROI. Defaults to 96.
        roi_height (int, optional): Height of the ROI. Defaults to 96.
        roi_width (int, optional): Width of the ROI. Defaults to 96.
        a_min (float, optional): Minimum intensity value for scaling. Defaults to -57.0.
        a_max (float, optional): Maximum intensity value for scaling. Defaults to 164.0.
        b_min (float, optional): Minimum output intensity value after scaling. Defaults to 0.0.
        b_max (float, optional): Maximum output intensity value after scaling. Defaults to 1.0.
        num_samples_per_image (int, optional): Number of samples to extract per image. Defaults to 1.
        transforms (Compose, optional): Custom transforms to apply. Defaults to None.

    Returns:
        DataLoader: A MONAI DataLoader instance for inference.
    """
    ids = []
    data_files = []
    files = os.listdir(data_path)
    for file in files:
        if not file.startswith('.'):
            img_id = file.split('_')[0].split('.nii.gz')[0]
            if img_id not in ids:
                ids.append(img_id)
                image_file = os.path.join(data_path, file)
                data_dict = {'image': image_file}
                if label_path is not None:
                    label_file = os.path.join(label_path, file)
                    if os.path.exists(label_file):
                        data_dict['label'] = label_file
                    else:
                        print(f"Label file {label_file} does not exist.")
                data_files.append(data_dict)

    if label_path is not None:
        keys = ["image", "label"]
    else:
        keys = ["image"]

    if transforms is None:
        transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=[1.5, 1.5, 2.0],
                mode=("bilinear", "nearest") if label_path else "bilinear"
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True
            ),
            EnsureTyped(keys=keys, track_meta=True),
        ])

    dataset = Dataset(data=data_files, transform=transforms)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader

def get_train_data_loader(
    data_path,
    label_path=None,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    roi_depth=96,
    roi_height=96,
    roi_width=96,
    a_min=-57.0,
    a_max=164.0,
    b_min=0.0,
    b_max=1.0,
    num_samples_per_image=2,
    transforms=None,
):
    ids = []
    data_files = []
    files = os.listdir(data_path)
    for file in files:
        if not file.startswith('.'):
            img_id = file.split('_')[0].split('.nii.gz')[0]
            if img_id not in ids:
                ids.append(img_id)
                image_file = os.path.join(data_path, file)
                data_dict = {'image': image_file}
                if label_path is not None:
                    label_file = os.path.join(label_path, file)
                    if os.path.exists(label_file):
                        data_dict['label'] = label_file
                    else:
                        print(f"Label file {label_file} does not exist.")
                data_files.append(data_dict)

    if label_path is not None:
        keys = ["image", "label"]
    else:
        keys = ["image"]

    if transforms is None:
        deterministic_transforms= [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=[1.5, 1.5, 2.0],
                mode=("bilinear", "nearest") if label_path else "bilinear"
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True
            ),
            EnsureTyped(keys=keys, track_meta=True),
        ]
        random_transforms = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(roi_width,roi_height,roi_depth),
                pos=1,
                neg=1,
                num_samples=num_samples_per_image,
                image_key="image",
                image_threshold=0
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.1
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.1
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.1
            ),
            RandRotate90d(
                keys=["image", "label"],
                max_k=3,
                prob=0.1
            ),
            RandShiftIntensityd(
                keys="image",
                offsets=0.1,
                prob=0.5
            ),
        ]
        transforms = Compose(
            deterministic_transforms + random_transforms
        )
            

    dataset = Dataset(data=data_files, transform=transforms)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader



def get_unest_data_loader(
    data_path,
    label_path=None,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    transforms=None,
):

    data_files = []
    files = os.listdir(data_path)
    for file in files:
        if not file.startswith('.') and file.endswith('.nii.gz'):
            file_name = file.split('.nii.gz')[0]
            
            image_file = os.path.join(data_path, file)
            data_dict = {'image': image_file}
            
            if label_path is not None:
                label_filename = file_name+'_seg.nii.gz'
                label_file = os.path.join(label_path, label_filename)
                
                if os.path.exists(label_file):
                    data_dict['label'] = label_file
                else:
                    print(f"Label file {label_file} does not exist.")
            
            data_files.append(data_dict)

    if label_path is not None:
        keys = ["image", "label"]
    else:
        keys = ["image"]

    if transforms is None:
        transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=keys, track_meta=True),
        ])

    dataset = Dataset(data=data_files, transform=transforms)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader

def get_unest_calibration_data_loader(
    data_path,
    label_path=None,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    transforms=None,
):

    data_files = []
    files = os.listdir(data_path)
    for file in files:
        if not file.startswith('.') and file.endswith('.nii.gz'):
            file_name = file.split('.nii.gz')[0]
            
            image_file = os.path.join(data_path, file)
            data_dict = {'image': image_file}
            
            if label_path is not None:
                label_filename = file_name+'_seg.nii.gz'
                label_file = os.path.join(label_path, label_filename)
                
                if os.path.exists(label_file):
                    data_dict['label'] = label_file
                else:
                    print(f"Label file {label_file} does not exist.")
            
            data_files.append(data_dict)

    if label_path is not None:
        keys = ["image", "label"]
    else:
        keys = ["image"]

    if transforms is None:
        transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            CenterSpatialCropd(
            keys=["image"],
            roi_size=(96,96,96)),
            EnsureTyped(keys=keys, track_meta=True),
        ])

    dataset = Dataset(data=data_files, transform=transforms)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader




def get_calibration_data_loader(
    data_path,
    label_path=None,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    roi_depth=96,
    roi_height=96,
    roi_width=96,
    a_min=-57.0,
    a_max=164.0,
    b_min=0.0,
    b_max=1.0,
    num_samples_per_image=1,
):
    """
    Creates a MONAI DataLoader for calibration data with training transforms.

    Args:
        data_path (str): Path to the directory containing the images.
        label_path (str, optional): Path to the directory containing the labels. Defaults to None.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.
        roi_depth (int, optional): Depth of the random cropped patches. Defaults to 96.
        roi_height (int, optional): Height of the random cropped patches. Defaults to 96.
        roi_width (int, optional): Width of the random cropped patches. Defaults to 96.
        a_min (float, optional): Minimum intensity value for scaling. Defaults to -57.0.
        a_max (float, optional): Maximum intensity value for scaling. Defaults to 164.0.
        b_min (float, optional): Minimum output intensity value after scaling. Defaults to 0.0.
        b_max (float, optional): Maximum output intensity value after scaling. Defaults to 1.0.
        num_samples_per_image (int, optional): Number of samples to extract per image. Defaults to 1.

    Returns:
        DataLoader: A MONAI DataLoader instance for calibration.
    """
    ids = []
    data_files = []
    files = os.listdir(data_path)
    for file in files:
        if not file.startswith('.'):
            img_id = file.split('_')[0].split('.nii.gz')[0]
            if img_id not in ids:
                ids.append(img_id)
                image_file = os.path.join(data_path, file)
                data_dict = {'image': image_file}
                if label_path is not None:
                    label_file = os.path.join(label_path, file)
                    if os.path.exists(label_file):
                        data_dict['label'] = label_file
                    else:
                        print(f"Label file {label_file} does not exist.")
                data_files.append(data_dict)

    if label_path is not None:
        keys = ["image", "label"]
    else:
        keys = ["image"]

    # Define Transforms
    transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=[1.5, 1.5, 2.0],
                mode=("bilinear", "nearest") if label_path else "bilinear"
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True
            ),
            RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=(roi_depth, roi_height, roi_width),
            pos=1,
            neg=1,
            num_samples=num_samples_per_image,
            image_key="image",
            image_threshold=0
            ),
            EnsureTyped(keys=keys, track_meta=True),
        ])

    dataset = Dataset(data=data_files, transform=transforms)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader

def dice(x, y):
    """
    Calculate the Dice coefficient between two binary masks.

    Args:
        x (numpy.ndarray): Predicted mask.
        y (numpy.ndarray): Ground truth mask.

    Returns:
        float: Dice coefficient.
    """
    intersect = np.sum(x * y)
    y_sum = np.sum(y)
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(x)
    return 2 * intersect / (x_sum + y_sum)
