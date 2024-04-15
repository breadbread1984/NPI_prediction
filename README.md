# Introduction

this project is to predict curve base on given N, I and P

# Usage

## generate dataset

```shell
python3 create_dataset.py --input <path to directory containing csv>
```

## train with linear regression

```shell
python3 train.py --input samples.pkl --output model.pkl
```

## generate curve

```shell
python3 eval.py --n <N> --i <I> --p <P> --format (csv|png)
```

