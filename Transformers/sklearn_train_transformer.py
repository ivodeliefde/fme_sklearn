﻿# ============================================================================
#
#  Name     : sklearn_train_transformer.py
#  
#  System   : FME Custom Transformer
#  
#  Language : Python
#  
#  Purpose  : Enabling FME users to implement Scikit-learn models inside a workspace.
# 
#        Copyright (c) 2020, Tensing. All rights reserved.
#
#   Redistribution and use of this sample code in source and binary forms, with 
#   or without modification, are permitted provided that the following 
#   conditions are met:
#   * Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice, 
#     this list of conditions and the following disclaimer in the documentation 
#     and/or other materials provided with the distribution.
# 
# ===============================================================================

import fmeobjects
from TransformerUtil import Transformer

try:
    from sklearn_object import sklearn_model
except:
    raise ImportError(
        "sklearn transformer not properly installed. Check https://github.com/ivodeliefde/fme_sklearn for installation instructions.")

try:
    import numpy as np
    import pandas as pd
except:
    raise ImportError(
        "Dependencies not installed. Run 'fme python -m pip install pandas scikit-learn --user' to resolve this error")


# ============================================================================
# Class to perform the overall logic of training a machine learning model.
class MachineLearningModelTrainer(Transformer):

    def __init__(self, instanceName, paramMap):
        Transformer.__init__(self, instanceName, paramMap)
        self.logger = fmeobjects.FMELogFile()
        self.logger.logMessageString("Transformer initialized", fmeobjects.FME_INFORM)
        self.sk = sklearn_model()
        self.target_variable = self.paramMap()["target_variable"]
        self.model_type = self.paramMap()["model_type"]
        self.model_architecture = self.paramMap()["model_architecture"]
        self.output_path = self.paramMap()["output_path"]
        self.sk.create_model(self.model_type, self.model_architecture)
        self.categorical_features = []
        self.numeric_features = []
        self.x = None
        self.y = None

    # Takes a feature and processes it
    def input(self, feature):

        # Collect data
        record = {}
        for n in feature.getAllAttributeNames():
            if n in ['_creation_instance', 'fme_feature_type', 'fme_geometry', 'fme_type']:
                continue
            record[n] = [feature.getAttribute(n)]

        target = record.pop(self.target_variable, None)
        pd_record = pd.DataFrame(data=record)
        pd_record = pd_record.select_dtypes(include=['float64', 'int64', 'bool', 'object'])

        # self.logger.logMessageString(f"Y: {target} {np.isnan(target) }\nX: {pd_record} {pd_record.isnull().values.any()}", fmeobjects.FME_INFORM)

        if self.y is None:
            self.y = np.array(target)
        else:
            self.y = np.append(self.y, target)

        if self.x is None:
            self.x = pd_record
            self.numeric_features = self.x.select_dtypes(include=['float64', 'int64']).columns.values
            self.categorical_features = self.x.select_dtypes(exclude=['float64', 'int64']).columns.values
            self.sk.create_pipeline(numeric_features=self.numeric_features, categorical_features=self.categorical_features)
        else:
            self.x = self.x.append(pd_record)

        # Send the feature on its way
        self.pyoutput(feature)

    def close(self):
        """
      Close the transformer by fitting the model and exporting it to a joblib file.
      """

        # Fit data to model
        self.logger.logMessageString(f"Fit Scikit-learn model with {self.x.shape[0]} records", fmeobjects.FME_INFORM)
        for c in self.numeric_features:
            self.x.loc[:,c] = pd.to_numeric(self.x.loc[:,c], errors='coerce')
        self.sk.pipe.fit(X=self.x, y=self.y)
        # Export model
        self.logger.logMessageString("Export Scikit-learn model", fmeobjects.FME_INFORM)
        self.sk.export_model(self.output_path)

        self.logger.logMessageString("Transformer closed", fmeobjects.FME_INFORM)
        del (self.logger)  # Needed to avoid "Not All FME sessions were destroyed"
