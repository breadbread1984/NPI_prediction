#!/usr/bin/python3

from shutil import rmtree
from os import mkdir
from os.path import exists, join
import pickle
from absl import flags, app
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = 'samples.pkl', help = 'path to training sampels')
  flags.DEFINE_string('output', default = 'model.pkl', help = 'directory for output')

def main(unused_argv):
  with open(FLAGS.input, 'rb') as f:
    samples = pickle.loads(f.read())
  X, Y = np.zeros((0,4)), np.zeros((0,))
  for (N,P,I), data in samples.items():
    data = np.array(data) # data.shape = (n,2)
    npi = np.tile(np.expand_dims(np.array([N,P,I]), axis = 0), (data.shape[0],1)) # npi.shape = (n, 3)
    x = np.concatenate([npi, data[:,:1]], axis = -1) # x.shape = (n, 4)
    y = data[:,1] # y.shape = (n, 1)
    X = np.concatenate([X, x], axis = 0)
    Y = np.concatenate([Y, y], axis = 0)
  print(X.shape, Y.shape)
  model = SVR(C = 1.0, epsilon = 0.2)
  model.fit(X,Y)
  with open(FLAGS.output,'wb') as f:
    f.write(pickle.dumps(model))

if __name__ == "__main__":
  add_options()
  app.run(main)

