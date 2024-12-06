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

def apply_affine_transformation(img, transform, reference_image):
    # Apply inverse affine transform to atlas
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(transform)
    transformed_segmentation = resampler.Execute(img)


    # segmentation = sitk.ReadImage(img)
    # transform = sitk.ReadTransform(transform)
    # transformed_segmentation = sitk.Resample(segmentation, reference_image, transform,
                                            #  sitk.sitkNearestNeighbor, 0, segmentation.GetPixelID())
    return transformed_segmentation

# Create atlas with multiple labels in separate channels
def create_multilabel_atlas(images, reference_image, label_transform, num_labels):
    atlas_shape = reference_image.GetSize()[::-1]  # Shape in z, y, x order
    accumulated_segmentation = np.zeros((*atlas_shape, num_labels), dtype=np.float32)

    for idx, img in enumerate(images):
        # Apply affine transformation and label transformation
        transformed_segmentation = apply_affine_transformation(img.images[structure.BrainImageTypes.GroundTruth], img.transformation, reference_image)
        transformed_segmentation = label_transform(transformed_segmentation)

        transformed_array = sitk.GetArrayFromImage(transformed_segmentation)

        # Accumulate each label in its own channel
        for label in range(1, num_labels + 1):
            label_mask = (transformed_array == label).astype(np.float32)
            accumulated_segmentation[..., label - 1] += label_mask

        print(f"Processed {idx + 1}/{len(images)} segmentations.")

    # Normalize to create probability maps
    num_segmentations = len(images)
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


def apply_opening_closing(atlas: sitk.Image, labels: list, kernel_radius: int = 2):
    """
    Perform morphological opening and closing on each label in the atlas.
    
    Args:
        atlas (sitk.Image): The atlas with labeled segmentations.
        labels (list): List of unique label values in the atlas.
        kernel_radius (int or list): Radius of the structuring element (default: 2).
    
    Returns:
        sitk.Image: Atlas with morphological opening and closing applied to each label.
    """
    print("Performing Opening and Closing.")

    # Create a copy of the atlas to apply modifications
    modified_atlas = sitk.Image(atlas.GetSize(), atlas.GetPixelID())
    modified_atlas.CopyInformation(atlas)
    
    # Convert kernel_radius to a tuple if it's an integer
    if isinstance(kernel_radius, int):
        kernel_radius = [kernel_radius] * atlas.GetDimension()
    
    for label in labels:
        if label == 0:
            # Skip the background label
            continue

        # Isolate the current label
        label_mask = sitk.BinaryThreshold(atlas, lowerThreshold=label, upperThreshold=label, insideValue=1, outsideValue=0)

        # Apply median filter before opening and closing
        radius_tuple = [1] * label_mask.GetDimension()
        filtered_mask = sitk.Median(label_mask, radius_tuple)

        # Perform morphological opening
        opened_mask = sitk.BinaryMorphologicalOpening(filtered_mask, kernelRadius=kernel_radius)

        # Perform morphological closing
        closed_mask = sitk.BinaryMorphologicalClosing(opened_mask, kernelRadius=kernel_radius)

        # Combine the processed label back into the atlas
        label_region = sitk.BinaryThreshold(closed_mask, lowerThreshold=1, upperThreshold=1, insideValue=label, outsideValue=0)
        modified_atlas = sitk.Add(modified_atlas, label_region)
    
    print("Opening and Closing finished.")
    return modified_atlas

def compute_weighted_atlas_label_specific(atlases, dice_scores_per_label, num_labels):
    """
    Create a weighted atlas using label-specific DICE scores.

    Parameters:
        atlases (list of sitk.Image): List of atlas images.
        dice_scores_per_label (dict): Dictionary with label-specific DICE scores for each atlas.
                                      Format: {label: [dice_score_atlas1, dice_score_atlas2, ...]}
        num_labels (int): Total number of labels in the segmentation.

    Returns:
        sitk.Image: Weighted atlas.
    """
    # Initialize weighted atlas with zeros
    atlas_shape = sitk.GetArrayFromImage(atlases[0]).shape
    weighted_atlas = np.zeros((num_labels + 1,) + atlas_shape, dtype=np.float32)

    # Process each label separately
    for label in range(1, num_labels + 1):  # Labels start from 1
        # Get DICE scores for the current label and normalize
        dice_scores = np.array(dice_scores_per_label[label])
        normalized_weights = dice_scores / np.sum(dice_scores)

        # Combine atlases for the current label
        label_contribution = np.zeros_like(weighted_atlas[0])
        for i, atlas in enumerate(atlases):
            atlas_array = sitk.GetArrayFromImage(atlas)
            label_mask = (atlas_array == label).astype(np.float32)  # Extract label
            label_contribution += label_mask * normalized_weights[i]

        # Add weighted contribution to the atlas
        weighted_atlas[label] = label_contribution

    # Combine all labels into a single probabilistic atlas
    combined_atlas = np.argmax(weighted_atlas, axis=0).astype(np.uint8)  # Multi-label output

    # Convert back to SimpleITK image
    combined_atlas_image = sitk.GetImageFromArray(combined_atlas)
    combined_atlas_image.CopyInformation(atlases[0])

    return combined_atlas_image

def register_atlas(
    t1_atlas: sitk.Image,
    t1_test: sitk.Image,
    registration,
    initial_transform=None
):
    """
    Register a T1w atlas to a T1w test image and apply the transformation to the labeled atlas.

    Parameters:
        t1_atlas (sitk.Image): T1-weighted atlas image.
        t1_test (sitk.Image): T1-weighted test image.

    Returns:
        sitk.Image: Labeled atlas aligned to the labeled test image.
    """
    # Ensure both images have the same pixel type
    t1_atlas = sitk.Cast(t1_atlas, sitk.sitkFloat32)
    t1_test = sitk.Cast(t1_test, sitk.sitkFloat32)

    # Resample moving image (t1_atlas) to match the fixed image (t1_test)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(t1_test)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())  # Identity transform
    t1_atlas_resampled = resampler.Execute(t1_atlas)
    # Step 1: Register T1w atlas to T1w test image
    registration_method = sitk.ImageRegistrationMethod()
    if registration == "affine":
        # Similarity metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=75)

        # Optimizer
        registration_method.SetOptimizerAsGradientDescent(learningRate=0.25, numberOfIterations=300)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Interpolation
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Initial transform
        initial_transform = sitk.CenteredTransformInitializer(
            t1_test,
            t1_atlas_resampled,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Multi-resolution strategy
        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    elif registration == "deformable":
        # Metric and optimizer setup
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=75)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.1)
        registration_method.SetInterpolator(sitk.sitkLinear)

        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.1,  # Try reducing learning rate if the registration is poor
            numberOfIterations=500,  # Increase iterations
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=50,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Add multi-resolution strategy
        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])  # Coarse-to-fine resolution
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])  # Add Gaussian smoothing
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Initial transform
        if initial_transform:
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
        else:
            initial_transform = sitk.CenteredTransformInitializer(
                t1_test, t1_atlas, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Perform registration
    final_transform = registration_method.Execute(t1_test, t1_atlas)

    print(f"Optimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f"Final metric value: {registration_method.GetMetricValue()}")

    return final_transform


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
    path_to_atlas = os.path.join(BASE_DIR, ATLAS_PATH)
    create_atlas = True
    if create_atlas:

        # initialize evaluator
        evaluator = putil.init_evaluator()

        # crawl the training image directories
        crawler = futil.FileSystemDataCrawler(data_train_dir,
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
        images_train = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True)
        print(len(images_train[:6]))
        # Example usage: list all unique values in each segmentation
        all_unique_labels = set()
        # if not sitk.ReadImage("brain_segmentation_label_atlas.nii"):
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
        num_of_atlases = 2

        for i in range(num_of_atlases):
            # Create the single-label atlas compatible with ITK-Snap
            label_atlas = create_multilabel_atlas(images_train[i*10:i*10+10], reference_image, label_transform, num_labels=len(to_combine))
            # Save the single-label atlas
            sitk.WriteImage(label_atlas, f"brain_segmentation_label_atlas_{i}.nii")

            # List of unique labels in the atlas
            labels = [1, 2, 3, 4, 5]  # Replace with actual labels in your atlas

            modified_atlas = apply_opening_closing(label_atlas, labels, kernel_radius = 1)
            # Save the single-label atlas
            sitk.WriteImage(modified_atlas, f"modified_brain_segmentation_label_atlas_{i}.nii")

            subimages = images_train[:i * 10] + images_train[i * 10 + 10:]

            for img in subimages:
                start_time = timeit.default_timer()
                print('-' * 10, 'Validation', img.id_)
                # Invert the transform
                inverse_transform = img.transformation.GetInverse()
                # Apply inverse affine transform to atlas
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(img.images[structure.BrainImageTypes.T1w])
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetTransform(inverse_transform)
                aligned_atlas = resampler.Execute(modified_atlas)
                print(' Time elapsed:', timeit.default_timer() - start_time, 's')

                # evaluate segmentation without post-processing
                evaluator.evaluate(aligned_atlas, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

            # use two writers to report the results
            os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
            folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(os.path.join(result_dir, folder_name), exist_ok=True)
            result_file = os.path.join(result_dir, folder_name, 'results.csv')
            writer.CSVWriter(result_file).write(evaluator.results)

            print('\nSubject-wise results...')
            writer.ConsoleWriter().write(evaluator.results)

            # report also mean and standard deviation among all subjects
            result_summary_file = os.path.join(result_dir, folder_name, 'results_summary.csv')
            functions = {'MEAN': np.mean, 'STD': np.std}
            writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
            print('\nAggregated statistic results...')
            writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

            # clear results such that the evaluator is ready for the next evaluation
            evaluator.clear()

    # load atlas images
    image_atlas = sitk.ReadImage(path_to_atlas)
    atlases = [sitk.ReadImage(f"modified_brain_segmentation_label_atlas_{i}.nii") for i in range(2)]  # Replace with actual paths
    dice_scores_per_label = {
    1: [0.5412968897962344, 0.5281777722628307],  # Grey matter
    2: [0.6856166796474245, 0.6832632058213017],  # White matter
    3: [0.6564597739817636, 0.6751220690580928],  # Hippocampus
    4: [0.6965166575751952, 0.8147284129367269],  # Amygdala
    5: [0.8235543067879816, 0.816336186]   # Thalamus
    }
    num_labels = 5
    weighted_atlas = compute_weighted_atlas_label_specific(atlases, dice_scores_per_label, num_labels)
    # Save the single-label atlas
    sitk.WriteImage(weighted_atlas, f"weighted_atlas.nii")

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
        print("Performing registration...")
        noisy_image = putil.add_salt_and_pepper_noise(img.images[structure.BrainImageTypes.T1w], salt_prob = 0.02, pepper_prob = 0.02)
        # Save the noisy image
        # sitk.WriteImage(noisy_image, f'noisy_{img.id_}.nii')
        transform = register_atlas(image_atlas, noisy_image, "affine")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img.images[structure.BrainImageTypes.T1w])
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(transform)
        aligned_atlas = resampler.Execute(weighted_atlas)

        
        print("Registration finished.")
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # evaluate segmentation without post-processing
        evaluator.evaluate(aligned_atlas, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(os.path.join(result_dir, folder_name), exist_ok=True)
    result_file = os.path.join(result_dir, folder_name, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, folder_name, 'results_summary.csv')
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
        default=os.path.normpath(os.path.join(script_dir, './mia-atlas-result')),
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
