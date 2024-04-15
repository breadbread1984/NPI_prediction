#!/usr/bin/python3

from os import listdir
from os.path import splitext, join, exists
import re
from absl import flags, app
import pickle
import numpy as np

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('sample', default = None, help = 'path to training samples')
  flags.DEFINE_string('model', default = 'model.pkl', help = 'path to model')
  flags.DEFINE_float('n', default = 0.25, help = 'N')
  flags.DEFINE_float('i', default = 0.25, help = 'I')
  flags.DEFINE_float('p', default = 0.005, help = 'P')
  flags.DEFINE_enum('format', default = 'csv', enum_values = {'csv', 'png'}, help = 'output format')

def main(unused_argv):
  with open(FLAGS.model, 'rb') as f:
    reg = pickle.loads(f.read())
  X = np.expand_dims(np.linspace(-2, 2, 41), axis = -1) # X.shape = (48, 1)
  npi = np.tile(np.expand_dims(np.array([FLAGS.n, FLAGS.p, FLAGS.i]), axis = 0), (X.shape[0],1)) # npi.shape = (48, 3,)
  inputs = np.concatenate([npi, X], axis = -1) # inputs.shape = (48, 4)
  Y = reg.predict(inputs)
  if FLAGS.format == 'csv':
    with open('results.csv', 'w') as f:
      for x,y in zip(X,Y):
        f.write('%f,%f\n' % (x,y))
  elif FLAGS.format == 'png':
    import matplotlib.pyplot as plt
    plt.plot(X, Y, label = 'prediction')
    if FLAGS.sample is not None:
      with open(FLAGS.sample,'rb') as f:
        samples = pickle.loads(f.read())
      xy = np.array(samples[(FLAGS.n, FLAGS.p, FLAGS.i)])
      plt.plot(xy[:,0], xy[:,1], label = 'ground truth')
    plt.legend()
    plt.savefig('results.png')

if __name__ == "__main__":
  add_options()
  app.run(main)

