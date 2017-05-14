# Multilayer Perceptron

[Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) based on [NumPy](http://www.numpy.org/)

Using [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) with [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to optimize network.

## Layers

* InputLayer
* SigmoidLayer
* ReLULayer
* LeakyReLULayer
* DropoutLayer

## Examples

See:

1. [Basic example](test.py)
2. [Auto encoder example](auto_encoder.py)
3. [Auto encoder & dropout example](auto_encoder_dropout.py)

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
