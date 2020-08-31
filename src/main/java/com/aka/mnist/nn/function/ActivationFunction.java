package com.aka.mnist.nn.function;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

public interface ActivationFunction {

    public double apply(double x);

    public double countDifferential(double x);
}
