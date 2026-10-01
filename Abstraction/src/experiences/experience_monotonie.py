import sys
import os
import time
import argparse

# Add the project root directory to PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)

import pandas as pd

from src.verification.boite import Boite
from src.verification.build_boite import BoitePropagator
from src.verification.monotonicity_checker import MonotonicityChecker
from src.experiences.verif_c_exemple import predict


"""
Usage:

python3 src/experiences/experience_monotonie.py \
    <folder_of_datasets> \
    <folder_of_models>

Example:

python3 src/experiences/experience_monotonie.py \
    Dataset \
    model

Optional output file:

python3 src/experiences/experience_monotonie.py \
    Dataset \
    model \
    -o resultats/monotonicity_results.txt
"""


# =============================================================
# COUNTEREXAMPLE
# =============================================================

def format_counterexample(model_path, c_exemple):
    """
    Format a monotonicity counterexample.

    The exact structure of c_exemple depends on the implementation
    of MonotonicityChecker. This function follows the structure used
    by the current implementation.
    """

    if c_exemple is None:
        return "No counterexample found."

    try:
        # Points stored in the counterexample
        x1 = c_exemple[0][0]
        x2 = c_exemple[0][1]
        x3 = c_exemple[1][0]
        x4 = c_exemple[1][1]

        # Predictions of the original model
        p1 = predict(model_path, x1)
        p2 = predict(model_path, x2)
        p3 = predict(model_path, x3)
        p4 = predict(model_path, x4)

        result = ""

        result += f"Point 1: {x1} -> prediction {p1}\n"
        result += f"Point 2: {x2} -> prediction {p2}\n"
        result += f"Point 3: {x3} -> prediction {p3}\n"
        result += f"Point 4: {x4} -> prediction {p4}\n"

        result += (
            "These points provide a counterexample to the "
            "considered monotonicity order."
        )

        return result

    except Exception as e:
        return (
            "A counterexample was found, but it could not be "
            f"formatted automatically: {c_exemple}\n"
            f"Formatting error: {e}"
        )


# =============================================================
# TEST ONE MODEL
# =============================================================

def tester_un_modele(
    dataset_path,
    model_path
):
    """
    Verify the monotonicity of one trained model.
    """

    print(
        "\n\n============================================"
    )

    print(
        "MONOTONICITY TEST"
    )

    print(
        "Dataset:",
        os.path.basename(dataset_path)
    )

    print(
        "Model:",
        os.path.basename(model_path)
    )

    print(
        "============================================\n"
    )

    start = time.time()

    # =========================================================
    # 1. INITIAL INPUT BOX
    # =========================================================

    boite_init = (
        Boite.creer_boite_initiale_depuis_dataset(
            dataset_path
        )
    )

    # =========================================================
    # 2. MODEL PROPAGATION
    # =========================================================

    propagateur = BoitePropagator(
        model_path,
        boite_init
    )

    resultats = propagateur.run()

    if not resultats:
        raise ValueError(
            f"No propagation result was produced for {model_path}."
        )

    # =========================================================
    # 3. DATASET INFORMATION
    # =========================================================

    df = pd.read_csv(
        dataset_path
    )

    # The last column is assumed to be the target label
    nb_features = (
        df.shape[1] - 1
    )

    # =========================================================
    # 4. FINAL BOXES
    # =========================================================

    final_boites = (
        BoitePropagator
        .regrouper_boites_par_classe(
            resultats[0]
        )
    )

    nb_boites = (
        propagateur.nb_boite
    )

    print(
        "Classes found:",
        list(final_boites.keys())
    )

    print(
        "Number of final boxes:",
        nb_boites
    )

    # =========================================================
    # 5. CLASS ORDER
    # =========================================================

    # Define the class order from the actual class identifiers.
    #
    # Example:
    # classes = [0, 1, 2]
    #
    # order = {
    #     0: 0,
    #     1: 1,
    #     2: 2
    # }
    #
    # This represents:
    #
    #     0 <= 1 <= 2

    classes = sorted(
        final_boites.keys()
    )

    order = {
        class_id: rank
        for rank, class_id
        in enumerate(classes)
    }

    print(
        "Class order:",
        order
    )

    # =========================================================
    # 6. MODEL SIZE
    # =========================================================

    model_size = (
        os.path.getsize(
            model_path
        )
        /
        1024.0
    )

    # =========================================================
    # 7. MONOTONICITY VERIFICATION
    # =========================================================

    monotonie_checker = MonotonicityChecker(
        final_boites,
        propagateur,
        order,
        model_path
    )

    is_monotone = (
        monotonie_checker.verif_monotone()
    )

    # =========================================================
    # 8. COUNTEREXAMPLE
    # =========================================================

    c_exemple = None
    counterexample_text = (
        "No counterexample found."
    )

    if monotonie_checker.c_exemple:

        c_exemple = (
            monotonie_checker.c_exemple
        )

        counterexample_text = (
            format_counterexample(
                model_path,
                c_exemple
            )
        )

    # =========================================================
    # 9. EXECUTION TIME
    # =========================================================

    end = time.time()

    execution_time = (
        end - start
    )

    # =========================================================
    # 10. DISPLAY RESULT
    # =========================================================

    print(
        "\n============================================"
    )

    print(
        "MONOTONICITY RESULT"
    )

    print(
        "============================================"
    )

    print(
        "Monotone:",
        is_monotone
    )

    print(
        "Number of features:",
        nb_features
    )

    print(
        "Number of boxes:",
        nb_boites
    )

    print(
        "Execution time:",
        f"{execution_time:.2f} s"
    )

    print(
        "Model size:",
        f"{model_size:.2f} KB"
    )

    if c_exemple is not None:

        print(
            "\nCounterexample:"
        )

        print(
            counterexample_text
        )

    print(
        "============================================\n"
    )

    # =========================================================
    # RESULT
    # =========================================================

    return {

        "dataset":
            os.path.basename(
                dataset_path
            ),

        "model":
            os.path.basename(
                model_path
            ),

        "monotone":
            is_monotone,

        "features":
            nb_features,

        "time_execution":
            execution_time,

        "boxes":
            nb_boites,

        "model_size_kb":
            round(
                model_size,
                2
            ),

        "class_order":
            order,

        "counterexample":
            c_exemple,

        "counterexample_text":
            counterexample_text,
    }


# =============================================================
# NORMALIZE FILENAMES
# =============================================================

def normaliser_nom(filename):
    """
    Normalize a filename in order to detect possible mismatches
    between datasets and models.
    """

    stem = os.path.splitext(
        filename
    )[0]

    return "".join(
        c.lower()
        for c in stem
        if c.isalnum()
    )


# =============================================================
# BATCH EXPERIMENT
# =============================================================

def experimentation_batch(
    dossier_datasets,
    dossier_models,
    chemin_resultat="resultats/monotonicity_results.txt"
):
    """
    Run the monotonicity experiment over all corresponding
    datasets and models.
    """

    fichiers_datasets = sorted(
        [
            f
            for f
            in os.listdir(
                dossier_datasets
            )
            if f.endswith(".csv")
        ]
    )

    fichiers_models = sorted(
        [
            f
            for f
            in os.listdir(
                dossier_models
            )
            if f.endswith(".json")
        ]
    )

    # =========================================================
    # CHECK NUMBER OF FILES
    # =========================================================

    if (
        len(fichiers_datasets)
        !=
        len(fichiers_models)
    ):

        raise ValueError(
            "The number of datasets and models is different: "
            f"{len(fichiers_datasets)} datasets "
            f"for {len(fichiers_models)} models."
        )

    # =========================================================
    # DISPLAY DATASET / MODEL PAIRS
    # =========================================================

    print(
        "\n=========================================="
    )

    print(
        "DATASET / MODEL PAIRS"
    )

    print(
        "=========================================="
    )

    for (
        dataset_file,
        model_file
    ) in zip(
        fichiers_datasets,
        fichiers_models
    ):

        print(
            dataset_file,
            "<---->",
            model_file
        )

        dataset_name = (
            normaliser_nom(
                dataset_file
            )
        )

        model_name = (
            normaliser_nom(
                model_file
            )
        )

        # Only display a warning.
        # The experiment is not interrupted.
        if (
            dataset_name not in model_name
            and
            model_name not in dataset_name
        ):

            print(
                "⚠️ WARNING: verify that this dataset "
                "corresponds to this model."
            )

    print(
        "==========================================\n"
    )

    # =========================================================
    # RUN EXPERIMENTS
    # =========================================================

    resultats = []

    for (
        dataset_file,
        model_file
    ) in zip(
        fichiers_datasets,
        fichiers_models
    ):

        dataset_path = (
            os.path.join(
                dossier_datasets,
                dataset_file
            )
        )

        model_path = (
            os.path.join(
                dossier_models,
                model_file
            )
        )

        resultat = (
            tester_un_modele(
                dataset_path,
                model_path
            )
        )

        resultats.append(
            resultat
        )

    # =========================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================

    output_directory = (
        os.path.dirname(
            chemin_resultat
        )
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # =========================================================
    # WRITE RESULTS
    # =========================================================

    with open(
        chemin_resultat,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "==== Results of the monotonicity experiment ====\n\n"
        )

        for r in resultats:

            f.write(
                f"Dataset : {r['dataset']}\n"
            )

            f.write(
                f"Model : {r['model']}\n"
            )

            f.write(
                "- Monotonicity : "
                f"{'YES' if r['monotone'] else 'NO'}\n"
            )

            f.write(
                "- Class order : "
                f"{r['class_order']}\n"
            )

            f.write(
                "- Number of features : "
                f"{r['features']}\n"
            )

            f.write(
                "- Number of boxes : "
                f"{r['boxes']}\n"
            )

            f.write(
                "- Execution time : "
                f"{r['time_execution']:.2f} s\n"
            )

            f.write(
                "- Model size : "
                f"{r['model_size_kb']} KB\n"
            )

            f.write(
                "- Counterexample:\n"
            )

            f.write(
                r["counterexample_text"]
            )

            f.write(
                "\n"
            )

            f.write(
                "-" * 60
                + "\n\n"
            )

    print(
        "\n✅ Results saved in",
        chemin_resultat
    )


# =============================================================
# COMMAND-LINE ARGUMENTS
# =============================================================

def parse_args(
    argv=None
):
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Verify the monotonicity of trained "
            "Gradient Boosting Decision Tree models."
        )
    )

    parser.add_argument(
        "datasets_dir",
        help=(
            "Folder containing the CSV datasets."
        )
    )

    parser.add_argument(
        "models_dir",
        help=(
            "Folder containing the JSON models."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        default="resultats/monotonicity_results.txt",
        help=(
            "Path to the result file "
            "(default: "
            "resultats/monotonicity_results.txt)"
        )
    )

    return parser.parse_args(
        argv
    )


# =============================================================
# MAIN
# =============================================================

def main(
    argv=None
):
    """
    Main entry point.
    """

    args = parse_args(
        argv
    )

    experimentation_batch(
        args.datasets_dir,
        args.models_dir,
        args.output
    )


if __name__ == "__main__":

    main()