# Visualize training history
import matplotlib.pyplot as plt
import numpy

def show_keys(history):
  print(history.history.keys())

def plot_accuracy(history):
  # summarize history for accuracy
  plt.clf()
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  plt.savefig("accuracy.png")

def plot_loss(history):  
  # summarize history for loss
  plt.clf()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  plt.savefig("loss.png")
