#============================================================================
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
#===============================================================================

import fmeobjects
from TransformerUtil import Transformer
from sklearn_object import sklearn_model
import numpy as np
import pandas as pd

#============================================================================
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
      self.logger.logMessageString("Feature type : "+feature.getAttribute("fme_feature_type"), fmeobjects.FME_INFORM)

      # Collect data
      record = {}
      for n in feature.getAllAttributeNames():
         if n in ['_creation_instance', 'fme_feature_type', 'fme_geometry', 'fme_type']:
            continue
         record[n] = [feature.getAttribute(n)]
      
      df = pd.DataFrame(data=record)
      pred = self.sk.model.predict(df)[0]
      feature.setAttribute("prediction",pred)

      # Send the feature on its way
      self.pyoutput(feature)
      self.logger.logMessageString("Feature processed", fmeobjects.FME_INFORM)

   def close(self):
      self.logger.logMessageString("Transformer closed", fmeobjects.FME_INFORM)
      del(self.logger) # Needed to avoid "Not All FME sessions were destroyed"
