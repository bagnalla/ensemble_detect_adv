# Training Ensembles to Detect Adversarial Examples

See the paper [here](https://arxiv.org/abs/1712.04006).

## Requisites

[Python 3](https://www.python.org/), [NumPy](http://www.numpy.org/), and [TensorFlow](https://www.tensorflow.org/)

## Easy setup

A makefile is included for training and evaluating the ensembles described in the paper. 

First train an ensemble:
* `make train DATASET=x` where x is either `mnist` or `cifar`

Then generate some adversarial examples targeting it:
* `make gen DATASET=x ATTACK=y` where y is `FGS`, `BI`, `DF`, `CW`, or `RAND`

Then evaluate it against the generated examples:
* `make eval DATASET=x`

Read below if you wish to experiment with different parameters.

## Training an ensemble

The file `train.py` can be used to train an ensemble from scratch.
Some important parameters:
* `-n, --ensemble_size` to set the number of ensemble members
* `--learning_rate` to set the initial learning rate
* `--eta` to set the eta parameter to control random perturbation
* `-d, --dataset` to choose between MNIST and CIFAR10
See the file for other parameters.

#### Example

```
python3 train.py -n 5 --dataset MNIST --learning_rate 0.1 --max_steps 100000 --eta 0.1 --model_dir models/myensemble
```

## Generating adversarial examples

The file `gen_adv.py` can be used to generate adversarial examples using the following methods:
* 0: [Fast gradient sign](https://arxiv.org/abs/1412.6572)
* 1: [Basic iterative](https://arxiv.org/abs/1607.02533)
* 2: [DeepFool](https://arxiv.org/abs/1511.04599)
* 3: [C&W l2](https://arxiv.org/abs/1608.04644)
* 4: Random noise

Use `-t` or `--type` to choose the attack method by its numeric index shown above.

Use `--direct` to save the adversarial examples directly in `adv_examples/`

See the file for other parameters.

#### Example
```
python3 gen_adv.py -n 5 --dataset MNIST --model_dir models/myensemble --attack 0 --epsilon 0.1
```

## Evaluating an ensemble

The file `eval.py` can be used to evaluate an ensemble's performance against both clean and adversarial examples.

Some parameters:
* `-rt, --rank_threshold` to set the detection parameter tau
* `-s, --set` to choose between the test and validation sets

See the file for other parameters.

#### Example

```
python3 eval.py -n 5 --dataset MNIST --model_dir models/myensemble --rank_threshold 2
```
