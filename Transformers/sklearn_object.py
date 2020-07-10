# -*- coding: utf-8 -*-
from sklearn import linear_model, tree, svm, cluster
from joblib import dump, load

class sklearn_model():
    """
    Scikit-learn model wrapper for the use in FME. 
    """

    def __init__(self):
        self.model_types = ["regression","classification", "clustering"]
        self.model_architectures = {"regression":["linear","tree","svm","sgd"], "classification": ["tree","svm","sgd"], "clustering": ["k-means"]}
        self.metrics = {"regression":["explained_variance", "mse"], "classification": ["accuracy","precision","f1score"], "clustering": []}
        self.model_type = None
        self.model_architecture = None
        self.metric = None
        self.model = None

    def available_models(self):
        s = ""
        for t in self.model_types:
            s += f"Model type: {t}\n"
            for a in self.model_architectures[t]:
                s += f" - {a}\n"
        return(s)

    def create_model(self, model_type, model_architecture, *args, **kwargs):
        """ 
        Initialise an Scikit-learn model. 

        Keyword arguments:
        model_type -- the kind of output the model should deliver (classification, regression or clustering are currently implemented)
        model_architecture -- the method the model should use (for example, linear, tree or svm)
        *args & **kwargs -- any additional arguments are directly passed on to the scikit-learn model. 
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
                self.model = linear_model.SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
            else:
                self.model = linear_model.SGDClassifier(*args, **kwargs)
        elif model_type == "clustering" and model_architecture == "k-means":
            if len(kwargs.items()) > 0 or len(args) > 0:
                self.model = cluster.KMeans(n_clusters=2, random_state=0)
            else:
                self.model = cluster.KMeans(*args, **kwargs)
        else:
            raise f"Combination of model type '{model_type}' and architecture '{model_architecture}' is invalid."

    def export_model(self, path="fme_model.joblib"):
        """
        Function to export the model to a joblib file.  
        """
        dump(self.model, f"{path}")
    
    def import_model(self, path="fme_model.joblib"):
        """
        Function to import a model from a joblib file.  
        """
        self.import_path = path
        self.model = load(path)

if __name__ == "__main__":
    fme_ml = sklearn_model()
    print(fme_ml.available_models())
    fme_ml.create_model("classification", "svm")
    print(fme_ml.model.get_params())
    print(fme_ml)