import os
import pytest
from .sklearn_object import sklearn_model
from sklearn import cluster

#####################################################
# Test initializing ML models
#####################################################
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


## Fail creating model test
def test_create_value_error():
    fme_ml = sklearn_model()
    with pytest.raises(ValueError) as excinfo:
        fme_ml.create_model("regression", "k-means")
        assert "Combination of model type 'regression' and architecture 'k-means' is invalid" in str(excinfo.value)

#####################################################
# Test I/O
#####################################################
def cleanup_io():
    """Remove test files if they already exist"""
    if os.path.isfile("fme_model.joblib"):
        os.remove("fme_model.joblib")
    if os.path.isfile("test.joblib"):
        os.remove("test.joblib")

cleanup_io()

def test_export_model():    
    """Check if export works"""
    fme_ml = sklearn_model()
    fme_ml.create_model("clustering", "k-means")
    fme_ml.export_model()
    assert os.path.isfile("fme_model.joblib")
    cleanup_io()
    
def test_overwrite_model():
    """Check if overwrite is possible"""
    fme_ml = sklearn_model()
    fme_ml.create_model("clustering", "k-means")
    fme_ml.export_model("test.joblib")
    assert os.path.isfile("test.joblib")
    fme_ml.export_model("test.joblib")
    assert os.path.isfile("test.joblib")
    cleanup_io()

def test_import_model():
    """Check if import works"""
    fme_ml1 = sklearn_model()
    fme_ml1.create_model("clustering", "k-means")
    fme_ml1.export_model("test.joblib")
    fme_ml2 = sklearn_model()
    fme_ml2.import_model("test.joblib")
    assert isinstance(fme_ml2.model, cluster.KMeans)
    cleanup_io()

#####################################################
# Test training a model
#####################################################

# TO DO!

#####################################################
# Test predicting with a model
#####################################################

# TO DO!

#####################################################
# Test evaluating a model
#####################################################

# TO DO!