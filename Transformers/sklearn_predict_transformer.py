# ============================================================================
#
#  Name     : sklearn_predict_transformer.py
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
    raise ImportError("sklearn transformer not properly installed. Check https://github.com/ivodeliefde/fme_sklearn for installation instructions.")

try:
    import numpy as np
    import pandas as pd
except:
    raise ImportError("Dependencies not installed. Run 'fme python -m pip install pandas scikit-learn --user' to resolve this error")


# ============================================================================
# Class to perform the overall logic of predicting using a machine learning model.
class MachineLearningModelTrainer(Transformer):

    def __init__(self, instanceName, paramMap):
        Transformer.__init__(self, instanceName, paramMap)
        self.logger = fmeobjects.FMELogFile()
        self.logger.logMessageString("Transformer initialized", fmeobjects.FME_INFORM)
        self.sk = sklearn_model()
        self.input_path = self.paramMap()["input_path"]
        self.sk.import_model(self.input_path)

    # Takes a feature and processes it
    def input(self, feature):
        # Collect data
        record = {}
        for n in feature.getAllAttributeNames():
            if n in ['_creation_instance', 'fme_feature_type', 'fme_geometry', 'fme_type']:
                continue
            record[n] = [feature.getAttribute(n)]

        pd_record = pd.DataFrame(data=record)
        for c in self.sk.numeric_features:
            pd_record.loc[:, c] = pd.to_numeric(pd_record.loc[:, c], errors='coerce')

        pred = self.sk.pipe.predict(pd_record)[0]
        feature.setAttribute("prediction",
                             pred)

        # Send the feature on its way
        self.pyoutput(feature)
        self.logger.logMessageString("Feature processed", fmeobjects.FME_INFORM)

    def close(self):
        self.logger.logMessageString("Transformer closed", fmeobjects.FME_INFORM)
        del (self.logger)  # Needed to avoid "Not All FME sessions were destroyed"
