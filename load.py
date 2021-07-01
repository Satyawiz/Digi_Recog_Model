from tensorflow.keras.models import model_from_json
import os
def init():
  cwd = os.getcwd()  # Get the current working directory (cwd)
  files = os.listdir(cwd)  # Get all the files in that directory
  # print("Files in %r: %s" % (cwd, files))
  json_file = open(cwd + '\\model\\model.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model.load_weights(cwd + "\\model\\mnist_model.h5")
  print("Loaded Model from disk")
  #compile and evaluate loaded model
  loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  return loaded_model