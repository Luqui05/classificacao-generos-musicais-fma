from __future__ import annotations

from collections import OrderedDict

from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def build_models() -> OrderedDict[str, BaseEstimator]:
    """
    Cria os classificadores utilizados no experimento.

    Todos os hiperparâmetros relevantes são definidos explicitamente
    para favorecer a reprodutibilidade do estudo.

    A normalização ocorre dentro de cada Pipeline. Dessa forma, o
    StandardScaler é ajustado somente sobre os dados de treinamento
    de cada fold, evitando vazamento de informação.

    Returns
    -------
    OrderedDict[str, BaseEstimator]
        Dicionário ordenado contendo o nome e o pipeline de cada modelo.
    """

    models: OrderedDict[str, BaseEstimator] = OrderedDict(
        {
            "Perceptron": Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        Perceptron(
                            max_iter=1_000,
                            tol=1e-3,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "SVM": Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            cache_size=2_048,
                            decision_function_shape="ovr",
                        ),
                    ),
                ]
            ),
            "KNN": Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        KNeighborsClassifier(
                            n_neighbors=5,
                            weights="uniform",
                            metric="minkowski",
                            p=2,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "Decision Tree": Pipeline(
                steps=[
                    (
                        "classifier",
                        DecisionTreeClassifier(
                            criterion="gini",
                            splitter="best",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Naive Bayes": Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        GaussianNB(
                            var_smoothing=1e-9,
                        ),
                    ),
                ]
            ),
        }
    )

    return models


def describe_models(
    models: OrderedDict[str, BaseEstimator],
) -> None:
    """
    Exibe no terminal os modelos e seus parâmetros principais.
    """

    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("-" * len(model_name))

        parameters = model.get_params(deep=True)

        classifier_parameters = {
            key: value
            for key, value in parameters.items()
            if key.startswith("classifier__")
        }

        for parameter_name, value in classifier_parameters.items():
            clean_name = parameter_name.replace("classifier__", "")
            print(f"{clean_name}: {value}")