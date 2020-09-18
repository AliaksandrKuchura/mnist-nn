package com.aka.mnist.nn.visitor;

import com.aka.mnist.nn.function.ActivationFunction;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.anyDouble;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

@ExtendWith(SpringExtension.class)
class ActivationFunctionVisitorTest {

    @Mock
    private ActivationFunction activationFunction;

    private ActivationFunctionVisitor visitor;

    @BeforeEach
    void setUp() {
        visitor = new ActivationFunctionVisitor(activationFunction);
    }

    @Test
    void visit() {
        double value = 1.0;
        Mockito.doReturn(value).when(activationFunction).apply(anyDouble());
        double actualResult = visitor.visit(0, 0, 0);

        assertEquals(value, actualResult);
    }
}
