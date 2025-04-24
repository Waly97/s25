import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def encode_and_save_train_test(train_path, test_path, target_column=None, output_dir="encoded_output"):
    """
    Encode proprement les données train/test avec gestion des NaN et des types.
    Sauvegarde les jeux encodés dans un dossier.
    """

    # 1. Charger les fichiers
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 2. Identifier les colonnes
    all_columns = df_train.columns.tolist()
    cat_columns = [col for col in all_columns if col != target_column and df_train[col].dtype == 'object']
    num_columns = [col for col in all_columns if col != target_column and df_train[col].dtype in ['float64', 'int64']]

    print(f"🔍 Colonnes catégorielles détectées : {cat_columns}")
    print(f"🔢 Colonnes numériques détectées : {num_columns}")

    # 3. Imputation des NaN
    # Catégorielles → "MISSING"
    df_train[cat_columns] = df_train[cat_columns].fillna("MISSING")
    df_test[cat_columns] = df_test[cat_columns].fillna("MISSING")

    # Numériques → moyenne (ou -999 si tu préfères)
    for col in num_columns:
        mean_val = df_train[col].mean()
        df_train[col].fillna(mean_val, inplace=True)
        df_test[col].fillna(mean_val, inplace=True)

    # 4. Fusion pour encodage cohérent
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    combined = pd.concat([df_train, df_test], axis=0)

    # 5. One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_data = encoder.fit_transform(combined[cat_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_columns))

    # 6. Fusion avec le reste des données
    combined = combined.drop(columns=cat_columns)
    combined = pd.concat([combined.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # 7. Encodage de la cible (LabelEncoder si nécessaire)
    if target_column:
        if combined[target_column].dtype == 'object':
            le = LabelEncoder()
            combined[target_column] = le.fit_transform(combined[target_column])
            print(f"🎯 Cible encodée : {list(le.classes_)}")

    # 8. Séparation train/test
    df_train_enc = combined[combined['is_train'] == 1].drop(columns='is_train')
    df_test_enc = combined[combined['is_train'] == 0].drop(columns='is_train')

    # 9. Sauvegarde
    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, os.path.basename(train_path).replace(".csv", "_encoded.csv"))
    test_out = os.path.join(output_dir, os.path.basename(test_path).replace(".csv", "_encoded.csv"))

    df_train_enc.to_csv(train_out, index=False)
    df_test_enc.to_csv(test_out, index=False)

    print(f"✅ Train encodé : {train_out}")
    print(f"✅ Test encodé  : {test_out}")

    return df_train_enc, df_test_enc

encode_and_save_train_test(
    train_path="datasets_train/wine_quility.csv",
    test_path="datasets_train/wine_quility.csv",
    target_column="output",
    output_dir="datasets_train_encoded"
)

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# import os

# def preprocess_csv_for_xgboost(input_csv, output_dir="datasets_train_encoded"):
#     """
#     Prépare un fichier CSV pour XGBoost : encodage + ajout d'une colonne cible.
#     La cible est ClaimNb / Exposure (fréquence sinistre).
#     """

#     # 1. Lecture du fichier CSV
#     df = pd.read_csv(input_csv)

#     # 2. Vérification des colonnes
#     num_features = ['BonusMalus', 'DrivAge', 'VehPower', 'VehAge', 'Density']
#     cat_features = ['Area', 'VehBrand', 'VehGas', 'Region']
#     features = num_features + cat_features

#     for col in features + ['ClaimNb', 'Exposure']:
#         if col not in df.columns:
#             raise ValueError(f"❌ Colonne manquante : {col}")

#     # 3. Création de la variable cible
#     df['target'] = df['ClaimNb'].astype(int)

#     # 4. Sélection des features uniquement
#     df_feat = df[features].copy()

#     # 5. Encodage de 'Area' (entier)
#     df_feat['Area'] = LabelEncoder().fit_transform(df_feat['Area'].astype(str))

#     print("✅ Features before one-hot-encoding:", df_feat.shape[1])

#     # 6. One-hot encoding des autres variables
#     df_feat = pd.get_dummies(df_feat, columns=['VehBrand', 'VehGas', 'Region'], drop_first=False)

#     # 7. Conversion des booléens en entiers
#     df_feat = df_feat.astype(int)

#     print("✅ Features after one-hot-encoding:", df_feat.shape[1])

#     # 8. Ajout des colonnes cibles + poids
#     df_feat['target'] = df['target']

#     # 9. Sauvegarde
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, os.path.basename(input_csv))
#     df_feat.to_csv(output_path, index=False)

#     print(f"📁 Fichier sauvegardé avec cible : {output_path}")


# preprocess_csv_for_xgboost("datasets_train/freq.csv")