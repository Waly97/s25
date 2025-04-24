import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
import numpy as np
import re

def convert_and_encode(input_file, target_column, output_dir="encoded_data"):
    """
    Convertit un fichier .data ou .csv en un fichier CSV avec One-Hot Encoding 
    pour les variables catégoriques (sauf la colonne cible qui est classée de 0 à n).
    
    :param input_file: Chemin du fichier .data ou .csv d'entrée
    :param target_column: Nom de la colonne cible
    :param output_dir: Répertoire où sauvegarder le fichier CSV encodé
    """

    # 1. Charger le fichier
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    # 2. Gérer les colonnes mixtes type "172N"
    for col in df.columns:
        if df[col].astype(str).str.match(r"^\d+[A-Za-z]$").all():
            print(f"🧪 Colonne '{col}' détectée comme mixte nombre + lettre.")
            df[col + "_num"] = df[col].astype(str).str.extract(r"^(\d+)[A-Za-z]$").astype(float)
            df[col + "_flag"] = df[col].astype(str).str.extract(r"^\d+([A-Za-z])$")
            df.drop(columns=[col], inplace=True)

    has_target = target_column in df.columns

    # 3. Identifier les colonnes numériques
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # 4. Identifier les colonnes catégoriques
    def is_categorical(column):
        return df[column].astype(str).str.contains(r"[a-zA-Z]", regex=True).any()

    categorical_features = [col for col in df.columns if col not in numeric_columns and is_categorical(col)]

    # 5. Retirer la colonne cible des features à encoder (si elle existe)
    if has_target and target_column in categorical_features:
        categorical_features.remove(target_column)

    if not categorical_features:
        print(f"⚠️ Aucune colonne catégorique détectée à encoder dans {input_file}.")
        categorical_encoded = pd.DataFrame()
    else:
        print(f"🔍 Colonnes catégoriques détectées : {categorical_features}")
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_data = encoder.fit_transform(df[categorical_features])
        column_names = encoder.get_feature_names_out(categorical_features)
        categorical_encoded = pd.DataFrame(encoded_data.astype(int), columns=column_names)

    # 6. Encodage de la cible (si elle est présente)
    if has_target:
        if is_categorical(target_column):
            label_encoder = LabelEncoder()
            df[target_column] = label_encoder.fit_transform(df[target_column])
            print(f"🎯 Cible '{target_column}' encodée : {list(label_encoder.classes_)}")

        # Vérification que les labels sont corrects
        unique_labels = df[target_column].unique()
        if not np.array_equal(np.sort(unique_labels), np.arange(len(unique_labels))):
            print(f"🚨 Problème : les labels doivent être dans [0, num_class - 1].")
            print(f"🔢 Labels trouvés : {unique_labels}")
            return

    # 7. Concaténation finale
    df_numeric = df[numeric_columns]
    df_final = pd.concat([df_numeric, categorical_encoded], axis=1)

    # Ajouter la colonne cible à la fin si elle existe
    if has_target:
        df_final[target_column] = df[target_column]

    # 8. Sauvegarde
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(input_file).replace(".data", ".csv")
    output_path = os.path.join(output_dir, output_filename)
    df_final.to_csv(output_path, index=False)

    if has_target:
        print(f"✅ Fichier encodé avec cible sauvegardé : {output_filename}")
    else:
        print(f"✅ Fichier encodé (sans cible) sauvegardé : {output_filename}")
def process_data_folder(input_folder, target_column, output_folder="encoded_data"):
    """
    Parcourt un dossier et applique `convert_and_encode` à tous les fichiers `.data` ou `.csv`.

    :param input_folder: Chemin du dossier contenant les fichiers .data
    :param target_column: Nom de la colonne cible
    :param output_folder: Dossier où sauvegarder les fichiers encodés
    """
    if not os.path.exists(input_folder):
        print("❌ Le dossier spécifié n'existe pas.")
        return

    #  Récupérer tous les fichiers .data ou .csv du dossier
    data_files = [f for f in os.listdir(input_folder) if f.endswith(".data") or f.endswith(".csv")]

    if not data_files:
        print("⚠️ Aucun fichier .data ou .csv trouvé dans le dossier.")
        return

    print(f"📂 {len(data_files)} fichiers trouvés : {data_files}")

    # Appliquer la conversion à chaque fichier
    for file in data_files:
        file_path = os.path.join(input_folder, file)
        convert_and_encode(file_path, target_column, output_folder)

# Exemple d'utilisation :
# Assurez-vous de remplacer "output" par le vrai nom de la colonne cible
process_data_folder("Abstraction/datasets_train", "output", "Abstraction/datasets_train_encoded")
