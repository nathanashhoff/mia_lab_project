# Brain Segmentation Atlas and Random Forest Classifier

This repository provides tools for generating a brain segmentation atlas and comparing its results with a Random Forest classifier model. The repository includes two main functionalities:

1. Generating a brain segmentation atlas.
2. Creating and evaluating a Random Forest classifier for brain segmentation.

## Requirements

Ensure you have Python installed. All required Python libraries are listed in `requirements.txt`. To install the dependencies, use:

```bash
pip install -r requirements.txt
```

## Usage

### Generating an Atlas

To generate a brain segmentation atlas, run the `atlas.py` script:

```bash
python atlas.py
```

This will create an atlas based on the provided data and save the results to the output directory.

### Generating the Random Forest Classifier

To create and evaluate the Random Forest classifier, run the `pipeline.py` script:

```bash
python pipeline.py
```

This script will train a Random Forest classifier on the provided dataset and compare its performance against the atlas results.

### Important Note

To run the Python files, one must be in the `mialab-main` folder.

### Generating the Paper

If the paper should be generated, please find the `.tex` file in the `report2` folder.

## Output

- **Atlas Results**: The generated atlas will be stored in the specified output directory.
- **Random Forest Classifier Results**: The classifier's performance metrics and model will be saved in the specified output directory.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvement.

