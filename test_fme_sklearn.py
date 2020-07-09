import os
from sklearn_object import sklearn_model
from sklearn import cluster
import pytest

## Regression tests
def test_create_regr_lin():
    fme_ml = sklearn_model()
    fme_ml.create_model("regression", "linear")
    assert fme_ml.model_type == "regression"
    assert fme_ml.model_architecture == "linear"

def test_create_regr_tree():
    fme_ml = sklearn_model()
    fme_ml.create_model("regression", "tree")
    assert fme_ml.model_type == "regression"
    assert fme_ml.model_architecture == "tree"

def test_create_regr_svm():
    fme_ml = sklearn_model()
    fme_ml.create_model("regression", "svm")
    assert fme_ml.model_type == "regression"
    assert fme_ml.model_architecture == "svm"

def test_create_regr_sgd():
    fme_ml = sklearn_model()
    fme_ml.create_model("regression", "sgd")
    assert fme_ml.model_type == "regression"
    assert fme_ml.model_architecture == "sgd"

## Classification tests
def test_create_class_tree():
    fme_ml = sklearn_model()
    fme_ml.create_model("classification", "tree")
    assert fme_ml.model_type == "classification"
    assert fme_ml.model_architecture == "tree"

def test_create_class_svm():
    fme_ml = sklearn_model()
    fme_ml.create_model("classification", "svm")
    assert fme_ml.model_type == "classification"
    assert fme_ml.model_architecture == "svm"

def test_create_class_sgd():
    fme_ml = sklearn_model()
    fme_ml.create_model("classification", "sgd")
    assert fme_ml.model_type == "classification"
    assert fme_ml.model_architecture == "sgd"

## Clustering tests
def test_create_cluster_kmeans():
    fme_ml = sklearn_model()
    fme_ml.create_model("clustering", "k-means")
    assert fme_ml.model_type == "clustering"
    assert fme_ml.model_architecture == "k-means"

## Import / Export tests
def test_export_model():
    # Remove test files if they already exist
    if os.path.isfile("fme_model.joblib"):
        os.remove("fme_model.joblib")
    if os.path.isfile("test.joblib"):
        os.remove("test.joblib")
    
    # Export model
    fme_ml = sklearn_model()
    fme_ml.create_model("clustering", "k-means")
    fme_ml.export_model()
    assert os.path.isfile("fme_model.joblib")
    fme_ml.export_model("test")
    assert os.path.isfile("test.joblib")

    # Check if overwrite is possible
    fme_ml.export_model("test")
    assert os.path.isfile("test.joblib")

def test_import_model():
    fme_ml1 = sklearn_model()
    fme_ml1.create_model("clustering", "k-means")
    fme_ml1.export_model("test")
    fme_ml2 = sklearn_model()
    fme_ml2.import_model("test.joblib")
    assert isinstance(fme_ml2.model, cluster.KMeans)