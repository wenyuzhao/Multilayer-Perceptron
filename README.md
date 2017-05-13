# Multilayer Perceptron

[Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) based on [NumPy](http://www.numpy.org/)

Supports **SigmoidLayer** / **ReLULayer** / **LeakyReLULayer**

## Examples

See:

1. [Basic example](test.py)
2. [Auto encoder example](auto_encoder.py)

## Auto encoder performance of [this example](auto_encoder.py)

```
input:
    ■
  ■ ■ ■ ■
■   ■   ■
    ■
    ■

prediction:
    ■
  ■ ■ ■
■   ■   ■
    ■
    ■

---------
input:
    ■
      ■
■ ■ ■ ■ ■
      ■ ■
    ■

prediction:
    ■
      ■
■ ■ ■ ■ ■
      ■
    ■

---------
input:
    ■
    ■
■   ■   ■
■ ■ ■ ■
    ■

prediction:
    ■
    ■
■   ■   ■
  ■ ■ ■
    ■

---------
input:
    ■
  ■
■ ■ ■ ■ ■
  ■
■   ■

prediction:
    ■
  ■
■ ■ ■ ■ ■
  ■
    ■
```
