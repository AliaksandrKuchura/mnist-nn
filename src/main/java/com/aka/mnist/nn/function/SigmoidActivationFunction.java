package com.aka.mnist.nn.function;

import org.springframework.stereotype.Component;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Component
public class SigmoidActivationFunction implements ActivationFunction {

    @Override
    public double apply(double x) {
        return sigmoid(x);
    }

    @Override
    public double applyDerivative(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
