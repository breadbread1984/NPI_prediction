#!/usr/bin/python3

from os import listdir
from os.path import splitext, join, exists
import re
from absl import flags, app
import pickle
import numpy as np

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('model', default = 'models', help = 'path to directory containing models')
  flags.DEFINE_float('n', default = 0.25, help = 'N')
  flags.DEFINE_float('i', default = 0.25, help = 'I')
  flags.DEFINE_float('p', default = 0.005, help = 'P')
  flags.DEFINE_enum('format', default = 'csv', enum_values = {'csv', 'png'}, help = 'output format')

def main(unused_argv):
  p =re.compile('n(.*)i(.*)p(.*)')
  combinations = list()
  for f in listdir(FLAGS.model):
    stem, ext = splitext(f)
    if ext != '.pkl': continue
    res = p.search(stem)
    N = float(res.group(1))
    I = float(res.group(2))
    P = float(res.group(3))
    combinations.append((N,I,P))
  if (FLAGS.n,FLAGS.i,FLAGS.p) not in combinations:
    raise Exception('unknown combination!')
  with open(join(FLAGS.model, 'n%.6fi%.6fp%.6f.pkl' % (FLAGS.n, FLAGS.i, FLAGS.p)), 'rb') as f:
    reg = pickle.loads(f.read())
  X = np.expand_dims(np.linspace(-2, 2, 41), axis = -1)
  Y = reg.predict(X)
  if FLAGS.format == 'csv':
    with open('results.csv', 'w') as f:
      for x,y in zip(X,Y):
        f.write('%f,%f\n' % (x,y))
  elif FLAGS.format == 'png':
    import matplotlib.pyplot as plt
    plt.plot(X,Y)
    plt.savefig('results.png')

if __name__ == "__main__":
  add_options()
  app.run(main)

