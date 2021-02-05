# helper for loading and saving a keras model
import os.path
import keras
from keras.models import Sequential
from keras.models import model_from_yaml

def loadModel(filename, weightsfilename):
  if os.path.exists(filename):
    print(F"Loading model {fileanme}")
    yaml_file = open(filename, 'r')
    model = model_from_yaml(yaml_file.read())
    yaml_file.close()
    model.load_weights(weightsfilename)
    retrun model
  else:
    print(F"No such file found: {filename}")

def saveModel(model, filename, weightsfilename):
  model_yaml = model.to_yaml()
  with open(filename, 'w') as yaml_file:
    yaml_file.write(model_yaml)
  model.save_weights(weightsfilename)


