import SimpleITK as sitk
import numpy as np
import os
import sys
import os
import matplotlib.pyplot as plt
import nibabel as nib

BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
ATLAS_PATH = "mia_lab_project/data/atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
FILE_PATH = 'mia_lab_project/data/train'
request_path = os.path.join(BASE_DIR, FILE_PATH)
path_to_atlas = os.path.join(BASE_DIR, ATLAS_PATH)

train_folders = os.listdir(request_path)
path_to_transform = os.path.join(request_path, train_folders[0], "affine.txt")
path_to_segmentation = os.path.join(request_path, train_folders[0], "labels_native.nii.gz")

class Transform:
    def __call__(self, img: sitk.Image) -> sitk.Image:
        pass

class ComposeTransform(Transform):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, img: sitk.Image) -> sitk.Image:
        for transform in self.transforms:
            img = transform(img)
        return img

class Resample(Transform):
    def __init__(self, new_spacing: tuple) -> None:
        super().__init__()
        self.new_spacing = new_spacing

    def __call__(self, img: sitk.Image) -> sitk.Image:
        size, spacing, origin, direction = img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection()
        scale = [ns / s for ns, s in zip(self.new_spacing, spacing)]
        new_size = [int(sz/sc) for sz, sc in zip(size, scale)]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputDirection(direction)
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(self.new_spacing)

        return resampler.Execute(img)

class MergeLabel(Transform):
    def __init__(self, to_combine: dict) -> None:
        super().__init__()
        self.to_combine = to_combine

    def __call__(self, img: sitk.Image) -> sitk.Image:
        np_img = sitk.GetArrayFromImage(img)
        merged_img = np.zeros_like(np_img)

        for new_label, labels_to_merge in self.to_combine.items():
            indices = np.reshape(np.in1d(np_img.ravel(), labels_to_merge, assume_unique=True), np_img.shape)
            merged_img[indices] = new_label

        out_img = sitk.GetImageFromArray(merged_img)
        out_img.CopyInformation(img)
        return out_img
    
# Function to create a 3D single-channel label map
def create_label_map_from_probabilities(probability_maps, reference_image, threshold=0.1):
    # Find the label with the highest probability at each voxel above a threshold
    label_map = np.argmax(probability_maps, axis=-1) + 1  # +1 to start labels from 1 instead of 0

    # Apply threshold to remove background noise
    max_probs = np.max(probability_maps, axis=-1)
    label_map[max_probs < threshold] = 0  # Set to background if below threshold

    # Convert the label map to a SimpleITK image
    label_map_image = sitk.GetImageFromArray(label_map.astype(np.uint8))
    label_map_image.CopyInformation(reference_image)

    return label_map_image

def apply_affine_transformation(segmentation_path, transform_path, reference_image):
    segmentation = sitk.ReadImage(segmentation_path)
    transform = sitk.ReadTransform(transform_path)
    transformed_segmentation = sitk.Resample(segmentation, reference_image, transform,
                                             sitk.sitkNearestNeighbor, 0, segmentation.GetPixelID())
    return transformed_segmentation

# Create atlas with multiple labels in separate channels
def create_multilabel_atlas(list_of_segmentations, list_of_transforms, reference_image, label_transform, num_labels):
    atlas_shape = reference_image.GetSize()[::-1]  # Shape in z, y, x order
    accumulated_segmentation = np.zeros((*atlas_shape, num_labels), dtype=np.float32)

    for idx, (segmentation_path, transform_path) in enumerate(zip(list_of_segmentations, list_of_transforms)):
        if not os.path.exists(segmentation_path) or not os.path.exists(transform_path):
            print(f"Skipping missing file: {segmentation_path} or {transform_path}")
            continue

        # Apply affine transformation and label transformation
        transformed_segmentation = apply_affine_transformation(segmentation_path, transform_path, reference_image)
        transformed_segmentation = label_transform(transformed_segmentation)

        transformed_array = sitk.GetArrayFromImage(transformed_segmentation)

        # Accumulate each label in its own channel
        for label in range(1, num_labels + 1):
            label_mask = (transformed_array == label).astype(np.float32)
            accumulated_segmentation[..., label - 1] += label_mask

        print(f"Processed {idx + 1}/{len(list_of_segmentations)} segmentations.")

    # Normalize to create probability maps
    num_segmentations = len(list_of_segmentations)
    probability_maps = accumulated_segmentation / num_segmentations

    # Check the unique values in probability maps for debugging
    print("Unique values in probability maps after accumulation and normalization:")
    for i in range(num_labels):
        print(f"Label {i + 1} - Min: {probability_maps[..., i].min()}, Max: {probability_maps[..., i].max()}")

    # Create a final 3D label map from the probability maps with thresholding
    label_map_image = create_label_map_from_probabilities(probability_maps, reference_image)

    return label_map_image

# Paths to data
list_of_segmentations = [os.path.join(request_path,folder, "labels_native.nii.gz") for folder in train_folders]  # Paths to segmentation masks
list_of_transforms = [os.path.join(request_path, folder, "affine.txt") for folder in train_folders]  # Paths to affine transformations

# Function to extract unique label values in a segmentation file
def find_unique_labels(segmentation_path):
    segmentation = sitk.ReadImage(segmentation_path)
    np_segmentation = sitk.GetArrayFromImage(segmentation)
    unique_labels = np.unique(np_segmentation)
    return unique_labels

# Example usage: list all unique values in each segmentation
all_unique_labels = set()

for file in list_of_segmentations:
    if not os.path.exists(file):
            print(f"Skipping missing file: {file}")
            continue
    unique_labels = find_unique_labels(file)
    print(f"Unique labels in {file}: {unique_labels}")
    all_unique_labels.update(unique_labels)

print(f"All unique labels across segmentations: {sorted(all_unique_labels)}")

# Define label merging dictionary
to_combine = {
    1: [1],  # Assuming label 1 corresponds to Grey Matter
    2: [2],  # White Matter
    3: [3],  # Hippocampus
    4: [4],  # Amygdala
    5: [5]   # Thalamus
}

# Define label transformation with resampling and label merging
label_transform = ComposeTransform([Resample((1., 1., 1.)), MergeLabel(to_combine)])


# Load the reference atlas image
reference_image = sitk.ReadImage(path_to_atlas)

# Create the single-label atlas compatible with ITK-Snap
label_atlas = create_multilabel_atlas(list_of_segmentations, list_of_transforms, reference_image, label_transform, num_labels=len(to_combine))

# Save the single-label atlas
sitk.WriteImage(label_atlas, "brain_segmentation_label_atlas.nii")