import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
import numpy as np

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
    :param target_column: Nom de la colonne cible à ne pas encoder en one-hot
    :param output_dir: Répertoire où sauvegarder le fichier CSV encodé
    """

    # 1. Charger le fichier
    df = pd.read_csv(input_file)

    # Normaliser les noms de colonnes (supprimer les espaces, convertir en minuscules)
    df.columns = df.columns.str.strip()

    # 1.1 Traiter les colonnes mixtes "numérique + lettre"
    for col in df.columns:
        if df[col].astype(str).str.match(r"^\d+[A-Za-z]$").all():
            print(f"🧪 Colonne '{col}' détectée comme mixte nombre + lettre.")
            
            # Séparer la partie numérique et la lettre dans deux nouvelles colonnes
            df[col + "_num"] = df[col].astype(str).str.extract(r"^(\d+)[A-Za-z]$").astype(float)
            df[col + "_flag"] = df[col].astype(str).str.extract(r"^\d+([A-Za-z])$")
            
            # Supprimer la colonne d'origine
            df.drop(columns=[col], inplace=True)

    # Vérifier si la colonne cible existe
    if target_column not in df.columns:
        print(f"⚠️ La colonne cible '{target_column}' n'existe pas dans {input_file}.")
        print(f"📌 Colonnes disponibles : {list(df.columns)}")  # Debug : Afficher les colonnes
        return

    # 2. Identifier les colonnes numériques (Ne pas encoder)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 3. Identifier les colonnes catégoriques (Seulement celles avec du texte)
    def is_categorical(column):
        return df[column].astype(str).str.contains(r'[a-zA-Z]', regex=True).any()

    categorical_features = [col for col in df.columns if col not in numeric_columns and is_categorical(col)]

    # Enlever la colonne cible des colonnes à encoder
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    if not categorical_features:
        print(f"Aucune colonne catégorique détectée pour {input_file}, sauvegarde sans modification.")
        output_filename = os.path.basename(input_file).replace(".data", ".csv")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, output_filename), index=False)
        print(f"📁 Fichier enregistré sous : {output_filename}")
        return

    print(f"🔍 Colonnes catégoriques détectées dans {input_file}: {categorical_features}")

    # 4. Appliquer One-Hot Encoding uniquement sur les colonnes catégoriques (sauf la cible)
    encoder = OneHotEncoder(sparse_output=False, drop="first")  # drop="first" pour éviter la multicolinéarité
    encoded_data = encoder.fit_transform(df[categorical_features])

    # 5. Convertir en DataFrame (avec entiers 0 et 1)
    column_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_data.astype(int), columns=column_names)

    # 6. Encodage de la colonne cible si elle est catégorique
    if is_categorical(target_column):
        label_encoder = LabelEncoder()
        df[target_column] = label_encoder.fit_transform(df[target_column])
        print(f"🎯 La colonne cible '{target_column}' a été encodée en valeurs entières : {list(label_encoder.classes_)}")

    # 7. Fusionner avec les colonnes numériques non modifiées et les colonnes encodées
    df_numeric = df[numeric_columns]  # Conserver uniquement les colonnes numériques
    final_df = pd.concat([df_numeric, encoded_df], axis=1)

    # 8. Assurer que la colonne cible est la dernière
    target_data = df[target_column]  # Extraire la colonne cible
    final_df[target_column] = target_data  # Ajouter la colonne cible à la fin

    # 9. Sauvegarde du fichier encodé
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(input_file).replace(".data", ".csv")
    final_df.to_csv(os.path.join(output_dir, output_filename), index=False)

    print(f"✅ Fichier encodé enregistré sous : {output_filename}")


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
process_data_folder("datasets_train", "output", "datasets_train_encoded")
