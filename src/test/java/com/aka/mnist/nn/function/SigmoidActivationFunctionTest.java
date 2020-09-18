package com.aka.mnist.nn.function;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

class SigmoidActivationFunctionTest {

    private SigmoidActivationFunction activationFunction = new SigmoidActivationFunction();

    private double[] inputs = new double[]{1.4, 10.3, 5.3, -1.1, -3.6};

    @Test
    void apply() {
        double[] expectedValues = new double[]{0.8021838, 0.999966368, 0.995033198, 0.249739894, 0.026596993};

        for (int i = 0; i < inputs.length; i++) {
            double actualValue = activationFunction.apply(inputs[i]);

            assertEquals(expectedValues[i], actualValue, 0.0000001);
        }
    }

    @Test
    void applyDifferential() {
        double[] expectedValues = new double[]{0.158684897, 0.000033630, 0.004942132, 0.187369879, 0.025889593};

        for (int i = 0; i < inputs.length; i++) {
            double actualValue = activationFunction.applyDerivative(inputs[i]);

            assertEquals(expectedValues[i], actualValue, 0.0000001);
        }
    }

    @Test
    void sigmoid() {
        double[] expectedValues = new double[]{0.8021838, 0.999966368, 0.995033198, 0.249739894, 0.026596993};

        for (int i = 0; i < inputs.length; i++) {
            double actualValue = activationFunction.sigmoid(inputs[i]);

            assertEquals(expectedValues[i], actualValue, 0.0000001);
        }
    }
}
