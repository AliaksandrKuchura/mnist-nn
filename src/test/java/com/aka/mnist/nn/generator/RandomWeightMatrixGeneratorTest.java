package com.aka.mnist.nn.generator;

import com.aka.mnist.config.Layer;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

@ExtendWith(SpringExtension.class)
class RandomWeightMatrixGeneratorTest {

    @Autowired
    ApplicationContext context;

    @MockBean
    private Supplier<Double> supplier;

    private RandomWeightMatrixGenerator generator;

    @BeforeEach
    void setup() {
        generator = new RandomWeightMatrixGenerator(supplier);
    }

    @Test
    void createListFromConfig() {
        double weight = 1.0;
        Mockito.doReturn(weight).when(supplier).get();
        List<Layer> layers = Arrays.asList(
                new Layer("test", 4),
                new Layer("test", 3),
                new Layer("test", 2)
        );
        List<RealMatrix> expectedWeightMatrices = Arrays.asList(
                MatrixUtils.createRealMatrix(new double[][]{
                        {weight, weight, weight, weight},
                        {weight, weight, weight, weight},
                        {weight, weight, weight, weight}
                }),
                MatrixUtils.createRealMatrix(new double[][]{
                        {weight, weight, weight},
                        {weight, weight, weight}
                })
        );
        List<RealMatrix> actualWeightMatrices = generator.createListFromConfig(layers);

        assertEquals(expectedWeightMatrices, actualWeightMatrices);
    }
}
