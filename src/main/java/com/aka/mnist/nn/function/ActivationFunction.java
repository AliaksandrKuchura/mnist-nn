package com.aka.mnist.nn.function;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

public interface ActivationFunction {

    double apply(double x);

    double applyDerivative(double x);
}
