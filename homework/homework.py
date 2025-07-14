import pandas as pd
import gzip
import json
import os
import pickle
import zipfile
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def extraer_csv(zipfile_path):
    with zipfile.ZipFile(zipfile_path, "r") as archivo:
        csv_name = archivo.namelist()[0]
        with archivo.open(csv_name) as contenido:
            return pd.read_csv(contenido)


def limpiar_datos(df):
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda val: 4 if val > 4 else val)
    return df


def crear_pipeline():
    categorias = ["SEX", "EDUCATION", "MARRIAGE"]
    numericas = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
        "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    transformador = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorias)
        ]
    )

    modelo = Pipeline([
        ("preproc", transformador),
        ("select", SelectKBest(score_func=f_classif, k=10)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    return modelo


def buscar_mejores_params(modelo, x_train, y_train):
    parametros = {
        "select__k": range(1, 11),
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "clf__solver": ["liblinear"],
    }

    grid = GridSearchCV(
        modelo, parametros, n_jobs=-1,
        cv=10, scoring="balanced_accuracy", refit=True
    )
    grid.fit(x_train, y_train)
    return grid


def guardar_modelo(modelo):
    with gzip.open("files/models/model.pkl.gz", "wb") as salida:
        pickle.dump(modelo, salida)


def calcular_metricas(modelo, x_train, y_train, x_test, y_test):
    pred_train = modelo.predict(x_train)
    pred_test = modelo.predict(x_test)

    m_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, pred_train)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, pred_train)),
        "recall": float(recall_score(y_train, pred_train)),
        "f1_score": float(f1_score(y_train, pred_train)),
    }
    m_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, pred_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
        "recall": float(recall_score(y_test, pred_test)),
        "f1_score": float(f1_score(y_test, pred_test)),
    }

    return m_train, m_test


def matrices_confusion(modelo, x_train, y_train, x_test, y_test):
    pred_train = modelo.predict(x_train)
    pred_test = modelo.predict(x_test)

    matriz_train = confusion_matrix(y_train, pred_train)
    matriz_test = confusion_matrix(y_test, pred_test)

    cm_train = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(matriz_train[0, 0]), "predicted_1": int(matriz_train[0, 1])},
        "true_1": {"predicted_0": int(matriz_train[1, 0]), "predicted_1": int(matriz_train[1, 1])},
    }
    cm_test = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(matriz_test[0, 0]), "predicted_1": int(matriz_test[0, 1])},
        "true_1": {"predicted_0": int(matriz_test[1, 0]), "predicted_1": int(matriz_test[1, 1])},
    }

    return cm_train, cm_test


def guardar_metricas(*metricas, archivo="files/output/metrics.json"):
    with open(archivo, "w") as salida:
        for fila in metricas:
            salida.write(json.dumps(fila) + "\n")


def crear_carpetas():
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)


if __name__ == "__main__":

    train_path = "files/input/train_data.csv.zip"
    test_path = "files/input/test_data.csv.zip"

    df_train = limpiar_datos(extraer_csv(train_path))
    df_test = limpiar_datos(extraer_csv(test_path))

    X_train = df_train.drop(columns=["default"])
    y_train = df_train["default"]
    X_test = df_test.drop(columns=["default"])
    y_test = df_test["default"]

    pipeline_modelo = crear_pipeline()
    modelo_final = buscar_mejores_params(pipeline_modelo, X_train, y_train)

    crear_carpetas()
    guardar_modelo(modelo_final)

    met_train, met_test = calcular_metricas(modelo_final, X_train, y_train, X_test, y_test)
    cm_train, cm_test = matrices_confusion(modelo_final, X_train, y_train, X_test, y_test)

    guardar_metricas(met_train, met_test, cm_train, cm_test)
