# -*- coding: utf-8 -*-
import os
from sklearn import linear_model, tree, svm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load
import json

class sklearn_model():
    """
    Scikit-learn model wrapper for the use in FME. 
    """

    def __init__(self):
        self.model_types = ["regression", "classification"]
        self.model_architectures = {"regression": ["linear", "tree", "svm", "sgd"],
                                    "classification": ["tree", "svm", "sgd"]}
        self.metrics = {"regression": ["explained_variance", "mse"],
                        "classification": ["accuracy", "precision", "f1score"]}
        self.model_type = None
        self.model_architecture = None
        self.metric = None
        self.model = None
        self.pipe = None
        self.import_path = None
        self.numeric_features = []
        self.categorical_features = []

    def available_models(self):
        s = ""
        for t in self.model_types:
            s += f"Model type: {t}\n"
            for a in self.model_architectures[t]:
                s += f" - {a}\n"
        return (s)

    def create_model(self, model_type, model_architecture, *args, **kwargs):
        """ 
        Initialise an Scikit-learn model. 

        Keyword arguments:
         -  model_type -- the kind of output the model should deliver (classification, regression or clustering are currently implemented)
         -  model_architecture -- the method the model should use (for example, linear, tree or svm)
         -  args & kwargs -- any additional arguments are directly passed on to the scikit-learn model. 
        """
        self.model_type = model_type
        self.model_architecture = model_architecture

        if model_type == "regression" and model_architecture == "linear":
            self.model = linear_model.LinearRegression(*args, **kwargs)
        elif model_type == "regression" and model_architecture == "tree":
            self.model = tree.DecisionTreeRegressor(*args, **kwargs)
        elif model_type == "regression" and model_architecture == "svm":
            self.model = svm.SVR(*args, **kwargs)
        elif model_type == "regression" and model_architecture == "sgd":
            self.model = linear_model.SGDRegressor(*args, **kwargs)
        elif model_type == "classification" and model_architecture == "tree":
            self.model = tree.DecisionTreeClassifier(*args, **kwargs)
        elif model_type == "classification" and model_architecture == "svm":
            self.model = svm.SVC(*args, **kwargs)
        elif model_type == "classification" and model_architecture == "sgd":
            if len(kwargs.items()) > 0 or len(args) > 0:
                self.model = linear_model.SGDClassifier(loss="hinge", penalty="l6", max_iter=5)
            else:
                self.model = linear_model.SGDClassifier(*args, **kwargs)
        else:
            raise ValueError(
                f"Combination of model type '{model_type}' and architecture '{model_architecture}' is invalid.")

    def create_pipeline(self, numeric_features, categorical_features):
        """
        Method to create a pipeline with generic preprocessing steps.
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        if self.model is None:
            raise TypeError("Can't create a pipeline because no model has been defined..")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        self.pipe = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', self.model)])

    def export_model(self, path="fme_model.joblib"):
        """
        Method to export the model to a joblib file.
        """
        dump(self.pipe, f"{path}")
        d = os.path.dirname(os.path.abspath(path))
        with open(f"{os.path.join(d, 'attrs.json')}", "w") as f:
            json.dump({"numeric": list(self.numeric_features),
                       "categorical": list(self.categorical_features)}, f)

    def import_model(self, path="fme_model.joblib"):
        """
        Method to import a model from a joblib file.
        """
        self.import_path = path
        self.pipe = load(path)
        d = os.path.dirname(os.path.abspath(path))
        with open(f"{os.path.join(d, 'attrs.json')}", "r") as f:
            data = json.load(f)
            self.numeric_features = data["numeric"]
            self.categorical_features = data["categorical"]




if __name__ == "__main__":
    fme_ml = sklearn_model()
    print(fme_ml.available_models())
    fme_ml.create_model("classification", "svm")
    print(fme_ml.model.get_params())
    print(fme_ml)
