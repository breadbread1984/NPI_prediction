#!/usr/bin/python3

from absl import flags, app
from os import listdir
from os.path import join, exists, splitext
from csv import reader
import pickle
import re

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'input directory')

def main(unused_argv):
  p = re.compile('\(IV_n(.*)p(.*)i(.*)\)')
  samples = dict()
  for f in listdir(FLAGS.input):
    stem, ext = splitext(f)
    if ext != '.csv': continue
    csv = open(join(FLAGS.input, f), 'r')
    keys = list()

    for idx, row in enumerate(csv.readlines()):
      row = row.split(',')
      if idx == 0:
        for i in range(len(row)//2):
          head = row[i * 2]
          res = p.search(head)
          n = float(res.group(1))
          p = float(res.group(2))
          i = float(res.group(3))
          samples[(n,p,i)] = list()
          keys.append((n,p,i))
      else:
        for i in range(len(row)//2):
          x = float(row[i * 2])
          y = float(row[i * 2 + 1])
          key = keys[i]
          samples[key].append((x,y))
  with open("samples.pkl", 'wb') as f:
    f.write(pickle.dumps(samples))

if __name__ == "__main__":
  add_options()
  app.run(main)

