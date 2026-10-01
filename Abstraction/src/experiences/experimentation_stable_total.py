import sys
import os
import time

# Ajouter la racine du projet au PYTHONPATH
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
from src.verification.stable_improve import StabilityChecker


"""
Usage:

python3 src/experiences/experientation_stable_total.py \
    <folder_of_datasets> \
    <folder_of_models>

Example:

python3 src/experiences/experientation_stable_total.py \
    Datasets \
    models
"""


# =============================================================
# EXTRACTION D'UNE INSTANCE D'UNE BOITE
# =============================================================

def extract_instance_from_boite(
    boite: Boite,
    mode="min"
):
    """
    Extract an instance from a box.

    mode = "min":
        lower corner

    mode = "max":
        upper corner
    """

    if mode == "min":

        return [
            bounds[0]
            for bounds
            in boite.bornes.values()
        ]

    elif mode == "max":

        return [
            bounds[1]
            for bounds
            in boite.bornes.values()
        ]

    else:

        raise ValueError(
            "mode must be 'min' or 'max'"
        )


# =============================================================
# AFFICHAGE CONTRE-EXEMPLE
# =============================================================

def printCE(c_exemple):

    if c_exemple is None:

        return "No counterexample found."

    class_id = c_exemple[0]

    broken = c_exemple[1]

    boite = broken[0]

    result = broken[1]

    fmin = extract_instance_from_boite(
        boite,
        "min"
    )

    fmax = extract_instance_from_boite(
        boite,
        "max"
    )

    between = extract_instance_from_boite(
        result["boite"],
        "min"
    )

    s1 = (
        "Instance min, a = "
        + str(
            (
                fmin,
                class_id
            )
        )
        + "\n"
    )

    s2 = (
        "Instance between, b = "
        + str(
            (
                between,
                result["prediction"]
            )
        )
        + "\n"
    )

    s3 = (
        "Instance max, c = "
        + str(
            (
                fmax,
                class_id
            )
        )
        + "\n"
    )

    s4 = "CONCLUSION\n"

    s5 = (
        "We have a <= b <= c, "
        "k(a) = k(c), and k(a) != k(b). "
        "Therefore, the model is not "
        "formally stable."
    )

    return (
        s1
        + s2
        + s3
        + s4
        + s5
    )


# =============================================================
# TEST D'UN MODELE
# =============================================================

def tester_un_modele(
    dataset_path,
    model_path
):
    """
    Test the stability of one model.
    """

    print(
        "\n\n========================================"
    )

    print(
        "Dataset:",
        os.path.basename(
            dataset_path
        )
    )

    print(
        "Model:",
        os.path.basename(
            model_path
        )
    )

    print(
        "========================================\n"
    )

    start = time.time()

    # ---------------------------------------------------------
    # Construction de la boîte initiale
    # ---------------------------------------------------------

    boite_init = (
        Boite.creer_boite_initiale_depuis_dataset(
            dataset_path
        )
    )

    # ---------------------------------------------------------
    # Charger / préparer le modèle
    # ---------------------------------------------------------

    propagateur = BoitePropagator(
        model_path,
        boite_init
    )

    # ---------------------------------------------------------
    # Propagation sur tous les arbres
    # ---------------------------------------------------------

    resultats = propagateur.run()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        dataset_path
    )

    # dernière colonne = label
    nb_features = (
        df.shape[1] - 1
    )

    # ---------------------------------------------------------
    # Boîtes finales regroupées par classe
    # ---------------------------------------------------------

    final_boites = (
        BoitePropagator
        .regrouper_boites_par_classe(
            resultats[0]
        )
    )

    # Nombre de boîtes
    nb_boites = (
        propagateur.nb_boite
    )

    # ---------------------------------------------------------
    # Taille du modèle
    # ---------------------------------------------------------

    model_size = (
        os.path.getsize(
            model_path
        )
        /
        1024
    )

    # ---------------------------------------------------------
    # Vérification stabilité
    # ---------------------------------------------------------

    stable_checker = StabilityChecker(
        final_boites,
        propagateur,
        model_path
    )

    (
        stable_90,
        _,
        _
    ) = stable_checker.verif_stable()

    end = time.time()

    # ---------------------------------------------------------
    # Premier contre-exemple
    # ---------------------------------------------------------

    c_exemple = None

    if stable_checker.contre_exemple:

        for (
            class_id,
            cex
        ) in (
            stable_checker
            .contre_exemple
            .items()
        ):

            if len(cex) > 0:

                c_exemple = (
                    class_id,
                    cex[0]
                )

                break

    # ---------------------------------------------------------
    # Résultat
    # ---------------------------------------------------------

    return {

        "dataset":
            os.path.basename(
                dataset_path
            ),

        "model":
            os.path.basename(
                model_path
            ),

        # Critère expérimental :
        # chaque classe >= 90 %
        "stable":
            stable_90,

        # Vraie stabilité formelle :
        # aucune violation
        "formal_stable":
            stable_checker.formal_stable,

        # Taux moyen
        "taux_stability":
            stable_checker.taux_stability,

        # Taux par classe
        "class_rates":
            stable_checker.class_stability_rates,

        "c_exemple":
            c_exemple,

        "features":
            nb_features,

        "time_execution":
            end - start,

        "boxes":
            nb_boites,

        "model_size_kb":
            round(
                model_size,
                2
            ),
    }


# =============================================================
# NORMALISATION DES NOMS
# =============================================================

def normaliser_nom(filename):
    """
    Used only to display a warning when a dataset and a model
    do not appear to correspond.
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
# EXPERIMENTATION SUR TOUS LES MODELES
# =============================================================

def experimentation_batch(
    dossier_datasets,
    dossier_models,
    chemin_resultat="resultats/stability_results.txt"
):
    """
    Run the stability experiment on all datasets and models.
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

    # ---------------------------------------------------------
    # Vérification du nombre de fichiers
    # ---------------------------------------------------------

    if (
        len(fichiers_datasets)
        !=
        len(fichiers_models)
    ):

        raise ValueError(
            "The number of datasets "
            "and models is different: "
            f"{len(fichiers_datasets)} datasets "
            f"for {len(fichiers_models)} models."
        )

    # ---------------------------------------------------------
    # Affichage des associations
    # ---------------------------------------------------------

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

        dname = normaliser_nom(
            dataset_file
        )

        mname = normaliser_nom(
            model_file
        )

        # Simple avertissement
        if (
            dname not in mname
            and
            mname not in dname
        ):

            print(
                "⚠️ WARNING: verify that these "
                "two files correspond."
            )

    print(
        "==========================================\n"
    )

    # ---------------------------------------------------------
    # Lancement des expériences
    # ---------------------------------------------------------

    resultats = []

    for (
        dataset_file,
        model_file
    ) in zip(
        fichiers_datasets,
        fichiers_models
    ):

        dataset_path = os.path.join(
            dossier_datasets,
            dataset_file
        )

        model_path = os.path.join(
            dossier_models,
            model_file
        )

        resultat = tester_un_modele(
            dataset_path,
            model_path
        )

        resultats.append(
            resultat
        )

    # ---------------------------------------------------------
    # Créer le dossier résultat si nécessaire
    # ---------------------------------------------------------

    result_dir = os.path.dirname(
        chemin_resultat
    )

    if result_dir:

        os.makedirs(
            result_dir,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # Ecriture du fichier résultat
    # ---------------------------------------------------------

    with open(
        chemin_resultat,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "==== Results of the experiment ====\n\n"
        )

        for r in resultats:

            f.write(
                f"Dataset : {r['dataset']}\n"
            )

            f.write(
                f"Model : {r['model']}\n"
            )

            # ---------------------------------------------
            # Stabilité formelle
            # ---------------------------------------------

            f.write(
                "- Formal stability : "
                f"{'YES' if r['formal_stable'] else 'NO'}\n"
            )

            # ---------------------------------------------
            # Critère expérimental 90 %
            # ---------------------------------------------

            f.write(
                "- Stability >= 90% for every class : "
                f"{'YES' if r['stable'] else 'NO'}\n"
            )

            # ---------------------------------------------
            # Taux moyen
            # ---------------------------------------------

            f.write(
                "- Average stability rate : "
                f"{r['taux_stability'] * 100:.2f}%\n"
            )

            # ---------------------------------------------
            # Taux par classe
            # ---------------------------------------------

            f.write(
                "- Stability rate by class :\n"
            )

            for (
                class_id,
                rate
            ) in r[
                "class_rates"
            ].items():

                f.write(
                    f"    Class {class_id}: "
                    f"{rate * 100:.2f}%\n"
                )

            # ---------------------------------------------
            # Informations modèle
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Contre-exemple
            # ---------------------------------------------

            f.write(
                "- Counterexample:\n"
            )

            f.write(
                printCE(
                    r["c_exemple"]
                )
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
# MAIN
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) < 3:

        print(
            "Usage:"
        )

        print(
            "python3 "
            "src/experiences/"
            "experientation_stable_total.py "
            "<dataset_folder> "
            "<model_folder>"
        )

        sys.exit(1)

    dossier_datasets = (
        sys.argv[1]
    )

    dossier_models = (
        sys.argv[2]
    )

    experimentation_batch(
        dossier_datasets,
        dossier_models
    )