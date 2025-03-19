import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np

def convert_and_encode(input_file, output_dir="encoded_data"):
    """
    Convertit un fichier .data ou .csv en un fichier CSV avec One-Hot Encoding 
    pour les variables catégoriques uniquement (sans toucher aux colonnes numériques)
    et enregistre le résultat dans un répertoire spécifié.

    :param input_file: Chemin du fichier .data ou .csv d'entrée
    :param output_dir: Répertoire où sauvegarder le fichier CSV encodé
    """

    # 1. Charger le fichier
    df = pd.read_csv(input_file)

    # 2. Identifier les colonnes numériques (Ne pas encoder)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 3. Identifier les colonnes catégoriques (Seulement celles avec du texte)
    def is_categorical(column):
        return df[column].astype(str).str.contains(r'[a-zA-Z]', regex=True).any()

    categorical_features = [col for col in df.columns if col not in numeric_columns and is_categorical(col)]

    if not categorical_features:
        print(f"Aucune colonne catégorique détectée pour {input_file}, sauvegarde sans modification.")
        output_filename = os.path.basename(input_file).replace(".data", ".csv")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, output_filename), index=False)
        print(f" Fichier enregistré sous : {output_filename}")
        return
    
    print(f"Colonnes catégoriques détectées dans {input_file}: {categorical_features}")

    # 4. Appliquer One-Hot Encoding uniquement sur les colonnes catégoriques
    encoder = OneHotEncoder(sparse_output=False, drop="first")  # drop="first" pour éviter la multicolinéarité
    encoded_data = encoder.fit_transform(df[categorical_features])

    # 5. Convertir en DataFrame (avec entiers 0 et 1)
    column_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_data.astype(int), columns=column_names)

    # 6. Fusionner avec les colonnes numériques non modifiées
    df_numeric = df[numeric_columns]  # Conserver uniquement les colonnes numériques
    final_df = pd.concat([df_numeric, encoded_df], axis=1)

    # 7. Sauvegarde du fichier encodé
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(input_file).replace(".data", ".csv")
    final_df.to_csv(os.path.join(output_dir, output_filename), index=False)

    print(f"✅ Fichier encodé enregistré sous : {output_filename}")

def process_data_folder(input_folder, output_folder="encoded_data"):
    """
    Parcourt un dossier et applique `convert_and_encode` à tous les fichiers `.data`.

    :param input_folder: Chemin du dossier contenant les fichiers .data
    :param output_folder: Dossier où sauvegarder les fichiers encodés
    """
    if not os.path.exists(input_folder):
        print(" Le dossier spécifié n'existe pas.")
        return

    #  Récupérer tous les fichiers .data du dossier
    data_files = [f for f in os.listdir(input_folder) if f.endswith(".data") or f.endswith(".csv")]

    if not data_files:
        print("Aucun fichier .data ou .csv trouvé dans le dossier.")
        return

    print(f" {len(data_files)} fichiers trouvés : {data_files}")

    # Appliquer la conversion à chaque fichier
    for file in data_files:
        file_path = os.path.join(input_folder, file)
        convert_and_encode(file_path, output_folder)

#  Exemple d'utilisation :
process_data_folder("datasets_raw", "datasets_encoded")
