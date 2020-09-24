package com.aka.mnist.nn.visitor;

import com.aka.mnist.nn.function.ActivationFunction;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

@Component
public class ActivationFunctionDerivativeVisitor implements RealMatrixChangingVisitor {

    private ActivationFunction activationFunction;

    @Autowired
    public ActivationFunctionDerivativeVisitor(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
        // NO SONAR
    }

    @Override
    public double visit(int row, int column, double value) {
        return activationFunction.applyDerivative(value);
    }

    @Override
    public double end() {
        return 0;
    }
}
