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


def train_random_forest(images):
    # Concatenate feature matrices and labels
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    # Initialize the random forest with appropriate parameters
    forest = sk_ensemble.RandomForestClassifier(max_features=data_train.shape[1], n_estimators=100, max_depth=5)

    # Train the model
    start_time = timeit.default_timer()
    forest.fit(data_train, labels_train)
    print('Training Time elapsed:', timeit.default_timer() - start_time, 's')

    return forest

def predict_with_random_forest(forest, images_test):
    predictions = []
    probabilities = []
    
    for img in images_test:
        # Predict labels for each test image
        preds = forest.predict(img.feature_matrix[0])
        probs = forest.predict_proba(img.feature_matrix[0])
        
        predictions.append(preds)
        probabilities.append(probs)
    
    return predictions, probabilities

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

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

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



############################### Flags to turn on/off ######################################################

    use_grid_search = False  # Set to False to skip grid search
    use_salt_and_pepper_noise_train = False  # Set to False to skip adding salt and pepper noise
    use_salt_and_pepper_noise_test = True  # Set to False to skip adding salt and pepper noise

###########################################################################################################

    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)


    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True)

    if use_salt_and_pepper_noise_train:
        for img in images:
            img.images[structure.BrainImageTypes.T1w] = putil.add_salt_and_pepper_noise(img.images[structure.BrainImageTypes.T1w], salt_prob=0.02, pepper_prob=0.02)
            img.images[structure.BrainImageTypes.T2w] = putil.add_salt_and_pepper_noise(img.images[structure.BrainImageTypes.T2w], salt_prob=0.02, pepper_prob=0.02)
    
    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    if use_grid_search:
        # Set up parameters for grid search
        param_grid = {
            'n_estimators': [500, 600, 700, 800, 900, 1000],  # Number of trees
            'max_depth': [35, 40, 45, 50],               # Maximum tree depth
            'max_features': ['sqrt', 'log2'],                # Number of features per split
        }

        #Best parameters salt pepper: {'max_depth': 45, 'max_features': 'sqrt', 'n_estimators': 700}

        # Initialize RandomForestClassifier with GridSearchCV
        forest = sk_ensemble.RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(forest, param_grid, cv=3, n_jobs=-1, scoring='f1_macro')

        # Fit model with grid search
        start_time = timeit.default_timer()
        grid_search.fit(data_train, labels_train)
        print('Grid Search Time elapsed:', timeit.default_timer() - start_time, 's')

        # Use the best estimator from grid search
        forest = grid_search.best_estimator_
        print('The best parameters are:', grid_search.best_params_)

    else:
        # Use only random forest with best current parameters if grid search is off
        forest = sk_ensemble.RandomForestClassifier(
            max_features=images[0].feature_matrix[0].shape[1],
            n_estimators=700,
            max_depth=45
        )

        # Fit the model without grid search
        start_time = timeit.default_timer()
        forest.fit(data_train, labels_train)
        print('Training Time elapsed:', timeit.default_timer() - start_time, 's')

    
    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True)

    if use_salt_and_pepper_noise_test:
        for img in images_test:
            img.images[structure.BrainImageTypes.T1w] = putil.add_salt_and_pepper_noise(img.images[structure.BrainImageTypes.T1w], salt_prob=0.02, pepper_prob=0.02)
            img.images[structure.BrainImageTypes.T2w] = putil.add_salt_and_pepper_noise(img.images[structure.BrainImageTypes.T2w], salt_prob=0.02, pepper_prob=0.02)
    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        predictions = forest.predict(img.feature_matrix[0])
        probabilities = forest.predict_proba(img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.nii'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.nii'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], os.path.join(result_dir, images_test[i].id_ + '_t1.nii'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], os.path.join(result_dir, images_test[i].id_ + '_t2.nii'), True)

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
