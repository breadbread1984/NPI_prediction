# Introduction

this project is to predict curve base on given N, I and P

# Usage

## generate dataset

```shell
python3 create_dataset.py --input <path to directory containing csv>
```

## train with linear regression

```shell
python3 train_linreg.py --input samples.pkl --output models
```

## generate curve

```shell
python3 gen_linreg.py --n <N> --i <I> --p <P>
```

