"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer
from sklearn.model_selection import GridSearchCV

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

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

# Function to extract unique label values in a segmentation file
def find_unique_labels(segmentation_path):
    segmentation = sitk.ReadImage(segmentation_path)
    np_segmentation = sitk.GetArrayFromImage(segmentation)
    unique_labels = np.unique(np_segmentation)
    return unique_labels


def compute_dice_score(seg1: sitk.Image, seg2: sitk.Image, label: int) -> float:
    """
    Compute the Dice Similarity Coefficient for a specific label.
    
    Args:
        seg1: First segmentation (aligned atlas).
        seg2: Second segmentation (ground truth).
        label: The label for which to compute the DICE score.
    
    Returns:
        Dice score as a float.
    """
    seg1_array = sitk.GetArrayFromImage(seg1)
    seg2_array = sitk.GetArrayFromImage(seg2)
    
    # Create binary masks for the specific label
    seg1_label = (seg1_array == label).astype(int)
    seg2_label = (seg2_array == label).astype(int)
    
    # Compute intersection and union
    intersection = np.sum(seg1_label * seg2_label)
    union = np.sum(seg1_label) + np.sum(seg2_label)
    
    # Avoid division by zero
    dice_score = 2 * intersection / union if union > 0 else 0.0
    return dice_score


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:
        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """
    BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
    ATLAS_PATH = "mia_lab_project/data/atlas/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
    FILE_PATH = 'mia_lab_project/data/train'
    request_path = os.path.join(BASE_DIR, FILE_PATH)
    path_to_atlas = os.path.join(BASE_DIR, ATLAS_PATH)

    train_folders = os.listdir(request_path)

    # Paths to data
    list_of_segmentations = [os.path.join(request_path,folder, "labels_native.nii.gz") for folder in train_folders]  # Paths to segmentation masks
    list_of_transforms = [os.path.join(request_path, folder, "affine.txt") for folder in train_folders]  # Paths to affine transformations

    # Example usage: list all unique values in each segmentation
    all_unique_labels = set()
    if not sitk.ReadImage("brain_segmentation_label_atlas.nii"):
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
            1: [1],  # Label 1 corresponds to Grey Matter
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

    # load atlas images
    image_atlas = sitk.ReadImage("brain_segmentation_label_atlas.nii")

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for testing and pre-process
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True)

    for img in images_test:
        start_time = timeit.default_timer()
        print('-' * 10, 'Testing', img.id_)
        # Invert the transform
        inverse_transform = img.transformation.GetInverse()
        # Apply inverse affine transform to atlas
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img.images[structure.BrainImageTypes.T1w])
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(inverse_transform)
        aligned_atlas = resampler.Execute(image_atlas)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # evaluate segmentation without post-processing
        evaluator.evaluate(aligned_atlas, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()
        

if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
