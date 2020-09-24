package com.aka.mnist.nn;

import com.aka.mnist.config.Layer;
import com.aka.mnist.config.PerceptronConfiguration;
import com.aka.mnist.dataloader.MnistImage;
import com.aka.mnist.nn.generator.WeightMatrixGenerator;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.doReturn;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

@ActiveProfiles("perceptron-test")
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = {PerceptronConfiguration.class, Layer.class})
class PerceptronNeuralNetworkTest {

    @Autowired
    private PerceptronConfiguration perceptronConfiguration;

    @MockBean
    private WeightMatrixGenerator weightMatrixGenerator;

    private RealMatrixChangingVisitor activationFunctionVisitor = new RealMatrixChangingVisitor() {
        @Override
        public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
        }

        @Override
        public double visit(int row, int column, double value) {
            return value * 2;
        }

        @Override
        public double end() {
            return 0;
        }
    };

    private RealMatrixChangingVisitor activationFunctionDerivativeVisitor = new RealMatrixChangingVisitor() {
        @Override
        public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
        }

        @Override
        public double visit(int row, int column, double value) {
            return value * 3;
        }

        @Override
        public double end() {
            return 0;
        }
    };

    private PerceptronNeuralNetwork neuralNetwork;

    @BeforeEach
    void setUp() {
        List<RealMatrix> matrices = Arrays.asList(
                MatrixUtils.createRealMatrix(new double[][]{
                        {1.0, 0.0, 0.5, 1.0},
                        {1.0, 0.5, 0.0, 0.0},
                        {0.5, 0.0, 0.5, 0.0}}),
                MatrixUtils.createRealMatrix(new double[][]{
                        {0.5, 0.0, 0.5},
                        {0.0, 1.0, 1.0}})
        );
        doReturn(matrices).when(weightMatrixGenerator).createListFromConfig(anyList());
        neuralNetwork = new PerceptronNeuralNetwork(perceptronConfiguration, weightMatrixGenerator,
                activationFunctionVisitor, activationFunctionDerivativeVisitor);
    }

    @Test
    void predict() {
        List<RealMatrix> matrices = Arrays.asList(
                MatrixUtils.createRealMatrix(new double[][]{
                        {1.0, -1.0, -0.5, 0.5},
                        {0.3, 0.5, -0.2, 0.4},
                        {0.1, 0.5, 0.3, -0.5}}),
                MatrixUtils.createRealMatrix(new double[][]{
                        {0.5, 0.0, 0.5},
                        {-0.5, 1.0, -0.5},
                        {0.5, 0.0, 0.5}})
        );
        doReturn(matrices).when(weightMatrixGenerator).createListFromConfig(anyList());
        PerceptronNeuralNetwork network = new PerceptronNeuralNetwork(perceptronConfiguration, weightMatrixGenerator,
                activationFunctionVisitor, activationFunctionDerivativeVisitor);

        MnistImage image = new MnistImage(new int[][]{{0, 120}, {255, 0}});

        double[] expectedPredict = new double[]{-0.87, 1.01, -0.87};
        double[] actualPredict = network.predict(image);

        assertArrayEquals(expectedPredict, actualPredict, 0.01);
    }

    @Test
    void train() {
        List<RealMatrix> matrices = Arrays.asList(
                MatrixUtils.createRealMatrix(new double[][]{
                        {0.3, 0.4, 0.5, 0.6},
                        {-0.1, -0.2, -0.3, -0.4},
                        {1.0, 0.0, 0.0, 1.0}}),
                MatrixUtils.createRealMatrix(new double[][]{
                        {-0.5, 0.0, -0.5},
                        {0.5, -1.0, 0.5},
                        {-0.5, 0.0, -0.5}})
        );
        doReturn(matrices).when(weightMatrixGenerator).createListFromConfig(anyList());
        PerceptronNeuralNetwork network = new PerceptronNeuralNetwork(perceptronConfiguration, weightMatrixGenerator,
                activationFunctionVisitor, activationFunctionDerivativeVisitor);

        List<MnistImage> inputImages = Collections.singletonList(new MnistImage(new int[][]{
                {0, 255},
                {255, 0}}));

        double actualAccuracy = network.train(inputImages);

        assertEquals(2.79, actualAccuracy, 0.01);
    }

    @Test
    void backPropagation() {
        List<RealMatrix> matrices = Arrays.asList(
                MatrixUtils.createRealMatrix(new double[][]{
                        {1.0, 0.5, 0.5, 1.0},
                        {1.0, 0.5, 0.5, 1.0},
                        {1.5, 1.0, 1.5, 1.0}}),
                MatrixUtils.createRealMatrix(new double[][]{
                        {0.5, 0.0, 0.5},
                        {0.5, 0.0, 0.5},
                        {0.0, 1.0, 1.0}})
        );
        doReturn(matrices).when(weightMatrixGenerator).createListFromConfig(anyList());
        PerceptronNeuralNetwork network = new PerceptronNeuralNetwork(perceptronConfiguration, weightMatrixGenerator,
                activationFunctionVisitor, activationFunctionDerivativeVisitor);

        List<RealMatrix> layerInputs = Arrays.asList(
                MatrixUtils.createColumnRealMatrix(new double[]{1, 0, 0, 1}),
                MatrixUtils.createColumnRealMatrix(new double[]{1, 1, 1}),
                MatrixUtils.createColumnRealMatrix(new double[]{0, 1, 0}));
        RealMatrix expectedOutput = MatrixUtils.createColumnRealMatrix(new double[]{1, 1, 0});

        network.backPropagation(layerInputs, expectedOutput);

        List<RealMatrix> expectedMatrices = Arrays.asList(
                MatrixUtils.createRealMatrix(new double[][]{
                        {0.7, 0.5, 0.5, 0.7},
                        {1.0, 0.5, 0.5, 1.0},
                        {1.2, 1.0, 1.5, 0.7}}),
                MatrixUtils.createRealMatrix(new double[][]{
                        {0.5, 0.0, 0.5},
                        {-0.1, -0.6, -0.1},
                        {0.0, 1.0, 1.0},})
        );

        assertEquals(expectedMatrices.size(), matrices.size());

        for (int i = 0; i < expectedMatrices.size(); i++) {
            RealMatrix actualMatrix = matrices.get(i);
            RealMatrix expectedMatrix = expectedMatrices.get(i);

            assertRealMatrixEquals(expectedMatrix, actualMatrix);
        }
    }

    @Test
    void findWeightDelta() {
        RealMatrix error = MatrixUtils.createColumnRealMatrix(new double[]{1.1, 2.2});
        RealMatrix input = MatrixUtils.createColumnRealMatrix(new double[]{3.3, 4.4, 5.5});

        RealMatrix expectedWeightDelta = MatrixUtils.createRealMatrix(new double[][]{
                {0.363, 0.484, 0.605},
                {0.726, 0.968, 1.21}
        });
        RealMatrix actualWeightDelta = neuralNetwork.findWeightDelta(error, input);

        assertRealMatrixEquals(expectedWeightDelta, actualWeightDelta);
    }

    @Test
    void applyVisitor() {
        RealMatrix input = MatrixUtils.createRealMatrix(new double[][]{
                {1.1, 2.2}, {3.3, 4.4}
        });

        RealMatrix expectedResult = MatrixUtils.createRealMatrix(new double[][]{
                {2.2, 4.4}, {6.6, 8.8}
        });
        RealMatrix actualResult = neuralNetwork.applyVisitor(input, activationFunctionVisitor);

        assertNotSame(expectedResult, actualResult);
        assertRealMatrixEquals(expectedResult, actualResult);
    }

    @Test
    void applyDerivativeVisitor() {
        RealMatrix input = MatrixUtils.createRealMatrix(new double[][]{
                {1.1, 2.2}, {3.3, 4.4}
        });

        RealMatrix expectedResult = MatrixUtils.createRealMatrix(new double[][]{
                {3.3, 6.6}, {9.9, 13.2}
        });
        RealMatrix actualResult = neuralNetwork.applyVisitor(input, activationFunctionDerivativeVisitor);

        assertNotSame(expectedResult, actualResult);
        assertRealMatrixEquals(expectedResult, actualResult);
    }

    @Test
    void testFindWeightDelta() {
        RealMatrix error = MatrixUtils.createColumnRealMatrix(new double[]{1.2, 2.3, 3.4, 4.5});
        RealMatrix input = MatrixUtils.createColumnRealMatrix(new double[]{5.6, 6.7, 7.8, 8.9, 9.1});

        RealMatrix expectedResult = MatrixUtils.createRealMatrix(new double[][]{
                {0.672, 0.804, 0.936, 1.068, 1.092},
                {1.288, 1.541, 1.794, 2.047, 2.093},
                {1.904, 2.278, 2.652, 3.026, 3.094},
                {2.52, 3.015, 3.51, 4.005, 4.095}});
        RealMatrix actualResult = neuralNetwork.findWeightDelta(error, input);

        assertRealMatrixEquals(expectedResult, actualResult);
    }


    @Test
    void calculateHiddenError() {
        RealMatrix error = MatrixUtils.createColumnRealMatrix(new double[]{1.2, 2.3});
        RealMatrix weightMatrix = MatrixUtils.createRealMatrix(new double[][]{
                {0.5, 1.5, 2.5, 3.7},
                {1.5, 2.7, 3.2, 5.2}});

        RealMatrix expectedResult = MatrixUtils.createColumnRealMatrix(new double[]{4.05, 8.01, 10.36, 16.4});
        RealMatrix actualResult = neuralNetwork.calculateHiddenError(weightMatrix, error);

        assertRealMatrixEquals(expectedResult, actualResult);
    }

    @Test
    void calculateOutputError() {
        RealMatrix eOutput = MatrixUtils.createColumnRealMatrix(new double[]{1.0, 2.5, 3.5, 4.2, 5.0});
        RealMatrix aOutput = MatrixUtils.createColumnRealMatrix(new double[]{0.5, 0.5, 2.5, 2.7, 3.0});
        RealMatrix inputDerivative = MatrixUtils.createColumnRealMatrix(new double[]{1.0, 2.0, 3.25, 4.0, 5.2});

        RealMatrix expectedResult = MatrixUtils.createColumnRealMatrix(new double[]{0.5, 4.0, 3.25, 6.0, 10.4});
        RealMatrix actualResult = neuralNetwork.calculateOutputError(eOutput, aOutput, inputDerivative);

        assertRealMatrixEquals(expectedResult, actualResult);
    }

    @Test
    void calculateInputs() {
        RealMatrix inputData = MatrixUtils.createColumnRealMatrix(new double[]{1.0, 2.0, 3.0, 4.0});
        List<RealMatrix> expectedResult = Arrays.asList(
                inputData,
                MatrixUtils.createColumnRealMatrix(new double[]{6.5, 2.0, 2.0}),
                MatrixUtils.createColumnRealMatrix(new double[]{8.5, 8.0}));

        List<RealMatrix> actualResult = neuralNetwork.calculateInputs(inputData);

        assertEquals(expectedResult.size(), actualResult.size());

        for (int i = 0; i < actualResult.size(); i++) {
            assertRealMatrixEquals(expectedResult.get(i), actualResult.get(i));
        }
    }

    @Test
    void checkLabel() {
        Exception exception = assertThrows(UnsupportedOperationException.class, () -> neuralNetwork.checkLabel((byte) 10));
        String expectedMsg = "Wrong input train label. Label should be in range 0..2. " +
                "Current value - 10. Check network configuration or input data";

        assertEquals(expectedMsg, exception.getMessage());
    }

    @Test
    void checkDataThrowException() {
        Exception exception = assertThrows(UnsupportedOperationException.class, () -> neuralNetwork.checkData(new int[]{0, 0, 1}));
        String expectedMsg = "Wrong input train data. Network has '4' inputs, but found '3'. " +
                "Check network configuration or input data";

        assertEquals(expectedMsg, exception.getMessage());
    }

    @Test
    void inputArrayFromData() {
        double[] expectedArray = new double[]{0.0, 1.0, 0.392156862, 0.196078431, 0.019607843};
        double[] actualArray = neuralNetwork.inputArrayFromData(new int[]{0, 255, 100, 50, 5});

        assertArrayEquals(expectedArray, actualArray, 0.000000001);
    }

    @Test
    void outArrayFromLabel() {
        double[] expectedArray = new double[]{0.0, 0.0, 1.0};
        double[] actualArray = neuralNetwork.outArrayFromLabel((byte) 2);

        assertArrayEquals(expectedArray, actualArray);
    }

    private void assertRealMatrixEquals(RealMatrix expectedResult, RealMatrix actualResult) {
        assertEquals(expectedResult.getRowDimension(), actualResult.getRowDimension());
        assertEquals(expectedResult.getColumnDimension(), actualResult.getColumnDimension());

        for (int row = 0; row < actualResult.getRowDimension(); row++) {
            assertArrayEquals(expectedResult.getRow(row), actualResult.getRow(row), 0.00000001);
        }
    }
}
