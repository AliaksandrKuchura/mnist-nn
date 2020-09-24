package com.aka.mnist.nn;

import com.aka.mnist.config.Layer;
import com.aka.mnist.config.PerceptronConfiguration;
import com.aka.mnist.dataloader.MnistImage;
import com.aka.mnist.nn.generator.WeightMatrixGenerator;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Component
public class PerceptronNeuralNetwork implements NeuralNetwork<double[], MnistImage> {

    private int inputDataSize;

    private int outDataSize;

    private double learningRate;

    private RealMatrixChangingVisitor activationFunctionDerivativeVisitor;

    private RealMatrixChangingVisitor activationFunctionVisitor;

    private List<RealMatrix> weightMatrices;

    @Autowired
    public PerceptronNeuralNetwork(PerceptronConfiguration perceptronConfiguration,
                                   WeightMatrixGenerator weightMatrixGenerator,
                                   RealMatrixChangingVisitor activationFunctionVisitor,
                                   RealMatrixChangingVisitor activationFunctionDerivativeVisitor) {
        int layers = perceptronConfiguration.countLayers();

        if (layers > 2) {
            List<Layer> layerConfigurations = perceptronConfiguration.getLayers();
            this.inputDataSize = layerConfigurations.get(0).getSize();
            this.outDataSize = layerConfigurations.get(layers - 1).getSize();
            this.activationFunctionVisitor = activationFunctionVisitor;
            this.activationFunctionDerivativeVisitor = activationFunctionDerivativeVisitor;
            this.learningRate = perceptronConfiguration.getLearningRate();
            this.weightMatrices = weightMatrixGenerator.createListFromConfig(layerConfigurations);
        } else {
            throw new UnsupportedOperationException("Neural network should have 2 or more layers." +
                    " Check perceptron configuration. ");
        }
    }

    @Override
    public double[] predict(MnistImage image) {
        int[] data = image.getDataAsVector();
        RealMatrix currentInput = MatrixUtils.createColumnRealMatrix(inputArrayFromData(data));
        RealMatrix layerOutput = null;

        for (RealMatrix weightMatrix : weightMatrices) {
            RealMatrix layerInput = weightMatrix.multiply(currentInput);
            layerOutput = applyVisitor(layerInput, activationFunctionVisitor);
            currentInput = layerOutput;
        }
        return layerOutput == null ? new double[0] : layerOutput.getColumn(0);
    }

    @Override
    public double train(List<MnistImage> trainingData) {
        double networkAccuracy = 1;

        for (int i = 0; i < trainingData.size(); i++) {
            MnistImage image = trainingData.get(i);
            int[] data = image.getDataAsVector();
            byte label = image.getLabel();
            checkData(data);
            checkLabel(label);
            RealMatrix inputData = MatrixUtils.createColumnRealMatrix(inputArrayFromData(data));
            RealMatrix expectedOutput = MatrixUtils.createColumnRealMatrix(outArrayFromLabel(label));
            List<RealMatrix> inputs = calculateInputs(inputData);
            backPropagation(inputs, expectedOutput);

            if (i == trainingData.size() - 1) {
                RealMatrix actualOutput = applyVisitor(inputs.get(inputs.size() - 1), activationFunctionVisitor);
                RealMatrix networkError = expectedOutput.copy().subtract(actualOutput);
                networkAccuracy = countNetworkAccuracy(networkError);
            }
        }
        return networkAccuracy;
    }

    void backPropagation(List<RealMatrix> layerInputs, RealMatrix expectedOutput) {
        RealMatrix error = null;
        RealMatrix previousWeightMatrix = null;
        int inputSize = layerInputs.size();

        for (int i = inputSize; i-- > 1; ) {
            RealMatrix input = layerInputs.get(i);
            RealMatrix previousLayerInput = layerInputs.get(i - 1).copy();
            previousLayerInput.walkInOptimizedOrder(activationFunctionVisitor);
            RealMatrix weightMatrix = weightMatrices.get(i - 1);
            RealMatrix actualOutput = applyVisitor(input, activationFunctionVisitor);
            RealMatrix inputDerivative = applyVisitor(input, activationFunctionDerivativeVisitor);


            if (i == inputSize - 1) {
                error = calculateOutputError(expectedOutput, actualOutput, inputDerivative);
            } else {
                error = calculateHiddenError(previousWeightMatrix, error);
            }
            RealMatrix weightDelta = findWeightDelta(error, previousLayerInput);
            previousWeightMatrix = weightMatrix;
            RealMatrix updatedWeightMatrix = weightMatrix.add(weightDelta);
            weightMatrices.set(i - 1, updatedWeightMatrix);
        }
    }

    RealMatrix calculateHiddenError(RealMatrix weightMatrix, RealMatrix error) {
        return weightMatrix.copy().transpose().multiply(error);
    }

    RealMatrix calculateOutputError(RealMatrix expectedOutput, RealMatrix actualOutput, RealMatrix inputDerivative) {
        RealMatrix subtract = expectedOutput.subtract(actualOutput);
        RealVector errorVector = subtract.getColumnVector(0)
                .ebeMultiply(inputDerivative.getColumnVector(0));
        return MatrixUtils.createColumnRealMatrix(errorVector.toArray());
    }

    RealMatrix findWeightDelta(RealMatrix error, RealMatrix previousLayerInput) {
        return error.multiply(previousLayerInput.copy().transpose()).scalarMultiply(learningRate);
    }

    RealMatrix applyVisitor(RealMatrix input, RealMatrixChangingVisitor activationFunctionVisitor) {
        RealMatrix output = input.copy();
        output.walkInOptimizedOrder(activationFunctionVisitor);
        return output;
    }

    private double countNetworkAccuracy(RealMatrix outError) {
        double[] data = outError.getColumn(0);
        float sum = 0;

        for (double v : data) {
            sum += Math.abs(v);
        }
        return sum / data.length;
    }

    List<RealMatrix> calculateInputs(RealMatrix inputData) {
        List<RealMatrix> inputs = new ArrayList<>();
        inputs.add(inputData);
        RealMatrix currentInput = inputData;

        for (RealMatrix weightMatrix : weightMatrices) {
            RealMatrix layerInput = weightMatrix.multiply(currentInput);
            RealMatrix layerOutput = applyVisitor(layerInput, activationFunctionVisitor);
            inputs.add(layerInput);
            currentInput = layerOutput;
        }
        return inputs;
    }

    void checkLabel(byte label) {
        if (0 > label || label > outDataSize - 1) {
            throw new UnsupportedOperationException(
                    String.format("Wrong input train label. Label should be in range 0..%d." +
                            " Current value - %d. Check network configuration or input data", outDataSize - 1, label));
        }
    }

    void checkData(int[] data) {
        if (data.length != inputDataSize) {
            throw new UnsupportedOperationException(
                    String.format("Wrong input train data. Network has '%d' inputs, but found '%d'." +
                            " Check network configuration or input data", inputDataSize, data.length));
        }
    }

    double[] inputArrayFromData(int[] data) {
        return Arrays.stream(data).mapToDouble(value -> (double) value / MnistImage.MAX_DATA_VALUE).toArray();
    }

    double[] outArrayFromLabel(byte label) {
        double[] outData = new double[outDataSize];
        outData[label] = 1.0;
        return outData;
    }
}
