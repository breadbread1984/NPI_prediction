#!/usr/bin/python3

from shutil import rmtree
from os import mkdir
from os.path import exists, join
import pickle
from absl import flags, app
import numpy as np
from sklearn.linear_model import RANSACRegressor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = 'samples.pkl', help = 'path to training sampels')
  flags.DEFINE_string('output', default = 'models', help = 'directory for output')

def main(unused_argv):
  if exists(FLAGS.output): rmtree(FLAGS.output)
  mkdir(FLAGS.output)
  with open(FLAGS.input, 'rb') as f:
    samples = pickle.loads(f.read())
  for (N,I,P), data in samples.items():
    data = np.array(data) # data.shape = (n,2)
    X = data[:,:1]
    Y = data[:,1:]
    reg = RANSACRegressor().fit(X,Y)
    print("(%f,%f,%f) regression score: %f" % (N,I,P,reg.score(X,Y)))
    with open(join(FLAGS.output, 'n%fi%fp%f.pkl' % (N,I,P)),'wb') as f:
      f.write(pickle.dumps(reg))

if __name__ == "__main__":
  add_options()
  app.run(main)

