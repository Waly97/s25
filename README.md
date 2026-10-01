# s25

This project aims to verify the stability and monotonicity of Gradient Boosting Decision Tree (GBDT) models, in particular models trained with XGBoost and LightGBM.

The implementation is written in Python.

# Supported Models

The current implementation supports models trained with:

- XGBoost;
- LightGBM.

If you want to analyze models produced by another machine-learning library,
you must adapt the implementation of the split operation to the structure of
the considered model.

The split implementation is located in:

```text
Abstraction/src/verification/boite
```

The corresponding split function must then be used in:

```text
Abstraction/src/verification/arbre
```

The current call is located around line 19.

# Dependencies

To run this project, Python 3 and several Python libraries are required.

The main dependencies are:

- XGBoost;
- LightGBM;
- pandas;
- NumPy;
- scikit-learn;
- tqdm;
- numba.

They can be installed using:

```bash
pip install xgboost lightgbm pandas numpy scikit-learn tqdm numba
```

XGBoost documentation:

https://xgboost.readthedocs.io/en/stable/install.html

LightGBM documentation:

https://lightgbm.readthedocs.io/en/stable/

# Implementation

The complete implementation of this project is written in Python.

The verification algorithms are located in:

```text
Abstraction/src/verification
```

The experimental scripts are located in:

```text
Abstraction/src/experiences
```

# Repository Structure

The main directories of the project are organized as follows:

```text
Abstraction/
│
├── src/
│   ├── verification/
│   │   ├── boite/
│   │   ├── arbre/
│   │   ├── stable_improve.py
│   │   ├── monotonicity_checker.py
│   │   ├── boite_model.py
│   │   └── ...
│   │
│   └── experiences/
│       ├── experience_monotonie.py
│       ├── experience_stabilite.py
│       ├── experience_one_hot.py
│       ├── experience_corrected_model.py
│       └── ...
│
├── Dataset/
├── models/
├── guest_datasets/
├── guest_models/
├── resultats/
└── models_box/
```

# Reproducing the Experiments

The scripts used to reproduce the experiments presented in the paper are
located in:

```text
Abstraction/src/experiences
```

The experiments include:

- stability verification;
- monotonicity verification;
- stability verification with one-hot encoded categorical features;
- construction and evaluation of corrected box-based classifiers.

The stability and monotonicity experiments take **folders** as arguments,
rather than a single model and dataset. Therefore, all datasets and models
contained in the specified folders are processed sequentially.

For example, if the folder contains the seven models used in the paper, the
verification is performed on all seven models during the same execution.

Because several models are analyzed and some of them generate a large number
of boxes, a complete batch experiment may take approximately **15 minutes or
more**, depending on the number of models, the number of trees, the number of
generated boxes, and the hardware used.

# Testing Your Own Models

If you want to test your own models, you first need to prepare both the
datasets and the trained models.

Categorical features must be encoded before training.

Two encodings can be used:

- ordinary numerical encoding, such as label encoding;
- one-hot encoding.

The encoded datasets can be placed in:

```text
Abstraction/guest_datasets
```

The trained models can be placed in:

```text
Abstraction/guest_models
```

The models must be trained using XGBoost or LightGBM and must use a tree-based
Gradient Boosting representation compatible with the current implementation.

# Stability Verification Without One-Hot Encoding

For datasets encoded using an ordinary numerical encoding, run:

```bash
python3 src/experiences/experience_stabilite.py \
    <dataset_folder> \
    <model_folder>
```

For example:

```bash
python3 src/experiences/experience_stabilite.py \
    guest_datasets \
    guest_models
```

The first argument is a folder containing the datasets and the second argument
is a folder containing the corresponding trained models.

The datasets and models are processed as pairs.

The default result file is:

```text
resultats/stability_results.txt
```

The result file contains, for each model:

- formal stability;
- the experimental stability criterion;
- the quantitative stability rate;
- stability rates for individual classes;
- the number of features;
- the number of generated boxes;
- verification time;
- model size;
- a counterexample when a stability violation is detected.

A typical result has the following form:

```text
Dataset : CPU.csv
Model : CPU.json
- Formal stability : NO
- Stability >= 90% for every class : YES
- Average stability rate : 99.80%
- Number of features : 6
- Number of boxes : 5306
- Execution time : ...
- Counterexample : ...
```

# Monotonicity Verification

To verify monotonicity, run:

```bash
python3 src/experiences/experience_monotonie.py \
    <dataset_folder> \
    <model_folder>
```

For example:

```bash
python3 src/experiences/experience_monotonie.py \
    guest_datasets \
    guest_models
```

As for stability verification, the command takes two folders and verifies all
corresponding dataset/model pairs.

The default result file is:

```text
resultats/monotonicity_results.txt
```

The result file contains, for each model:

- whether the model is monotone;
- the considered class order;
- the number of features;
- the number of generated boxes;
- verification time;
- model size;
- a counterexample when monotonicity is violated.

Since all models in the folders are analyzed sequentially, a complete
monotonicity experiment over several models may take approximately
**15 minutes or more**, depending on their complexity.

# Stability Verification With One-Hot Encoding

For datasets whose categorical features are represented using one-hot
encoding, run:

```bash
python3 src/experiences/experience_one_hot.py \
    <dataset_folder> \
    <model_folder>
```

For example:

```bash
python3 src/experiences/experience_one_hot.py \
    guest_datasets \
    guest_models
```

The one-hot verification procedure first detects the groups of one-hot
features and considers only valid categorical configurations.

The default result file is:

```text
resultats_one_hot.txt
```

An alternative output file can be specified using:

```bash
python3 src/experiences/experience_one_hot.py \
    guest_datasets \
    guest_models \
    -o resultats/my_one_hot_results.txt
```

The result file reports:

- formal stability;
- the stability criterion for each valid one-hot configuration;
- quantitative stability rates;
- stability rates by class;
- number of valid one-hot configurations;
- number of generated boxes;
- execution time;
- model size;
- possible counterexamples.

# Corrected Box-Based Model Experiment

The project also provides an experiment for generating a corrected box-based
classifier.

The corresponding script is:

```text
src/experiences/experience_corrected_model.py
```

Unlike the previous batch experiments, this experiment takes **one model and
one dataset** as input.

The procedure performs the following steps:

1. the original model is verified for stability;
2. the quantitative stability criterion is checked;
3. if the required stability criterion is satisfied, the intermediate boxes
   obtained during verification are used to construct a corrected box-based
   classifier;
4. the corrected classifier is saved as a JSON file;
5. the accuracy of the original model is computed on the dataset;
6. the accuracy of the corrected box-based model is computed on the same
   dataset;
7. the sizes of the two models are compared;
8. a text report containing the complete comparison is generated.

To run the experiment:

```bash
python3 src/experiences/experience_corrected_model.py \
    <model.json> \
    <dataset.csv>
```

For example:

```bash
python3 src/experiences/experience_corrected_model.py \
    models/CPU.json \
    Dataset/CPU.csv
```

By default, the results are stored in:

```text
models_box/
```

A separate directory is automatically created for each model.

For example, for `CPU.json`, the generated structure is:

```text
models_box/
└── CPU/
    ├── CPU_box.json
    └── CPU_comparison.txt
```

The file:

```text
CPU_box.json
```

contains the corrected box-based classifier.

The file:

```text
CPU_comparison.txt
```

contains the experimental comparison between the original model and the
corrected model.

The report contains information such as:

```text
STABILITY
- Formal stability of original model
- Experimental stability criterion
- Average stability rate
- Stability rate by class

ACCURACY
- Original model accuracy
- Corrected model accuracy
- Accuracy difference

MODEL SIZE
- Original model size
- Corrected model size
- Size reduction

EXECUTION
- Total execution time
```

For example:

```text
Original model accuracy: 99.80%
Corrected model accuracy: 99.80%

Original model size: 162.48 KB
Corrected model size: 12.00 KB
```

A different root output directory can be specified using:

```bash
python3 src/experiences/experience_corrected_model.py \
    models/CPU.json \
    Dataset/CPU.csv \
    -o my_corrected_models
```

The result will then be stored in:

```text
my_corrected_models/CPU/
```

# Expected Results

Depending on the selected experiment, the implementation reports information
related to:

- formal stability;
- quantitative stability;
- model monotonicity;
- number of generated boxes;
- class bounding boxes;
- verification time;
- counterexamples;
- corrected box-based classifiers;
- prediction accuracy;
- model size.

The stability verification analyzes regions induced by the decision trees
rather than only the finite instances contained in the original dataset.

The monotonicity verification checks whether the classifier satisfies the
considered class order over the generated regions of the input space.

The corrected-model experiment uses the regions obtained during stability
verification to construct a new box-based classifier and compare it with the
original GBDT model.

# Output Files Summary

The main experiment outputs are:

```text
Stability:
resultats/stability_results.txt

Monotonicity:
resultats/monotonicity_results.txt

One-hot stability:
resultats_one_hot.txt

Corrected models:
models_box/<model_name>/<model_name>_box.json

Corrected-model comparison:
models_box/<model_name>/<model_name>_comparison.txt
```

# Execution Time

The stability and monotonicity experiments receive folders containing several
datasets and models.

Therefore, one command may execute verification for multiple models.

For the models used in the paper, a complete experiment may take approximately
**15 minutes or more**. The exact execution time depends mainly on:

- the number of models;
- the number of trees;
- tree depth;
- the number of generated boxes;
- the number of features;
- the hardware used.

Models generating a very large number of boxes may require substantially more
time than smaller models.

# Using Other Tree-Based Models

The current implementation directly supports XGBoost and LightGBM.

If you want to verify a model generated using another tree-based
machine-learning library, the representation of the tree splits must first be
adapted to the structure of that model.

The relevant split implementation is located in:

```text
Abstraction/src/verification/boite
```

The adapted split function must then be called from:

```text
Abstraction/src/verification/arbre
```

The current call is located around line 19.

# Source Code

All source files required for verification and experimentation are located
under:

```text
Abstraction/src
```