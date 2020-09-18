package com.aka.mnist.nn.generator;

import com.aka.mnist.config.Layer;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

@Component
public class RandomWeightMatrixGenerator implements WeightMatrixGenerator {

    private Supplier<Double> weightSupplier;

    @Autowired
    public RandomWeightMatrixGenerator(Supplier<Double> weightSupplier) {
        this.weightSupplier = weightSupplier;
    }

    @Override
    public List<RealMatrix> createListFromConfig(List<Layer> layerConfigList) {
        int layers = layerConfigList.size();
        List<RealMatrix> matrices = new ArrayList<>(layers - 1);
        int previousLayerSize = 0;

        for (int i = 0; i < layers; i++) {
            Integer layerSize = layerConfigList.get(i).getSize();

            if (i != 0) {
                double[][] values = initArray(layerSize, previousLayerSize);
                RealMatrix realMatrix = MatrixUtils.createRealMatrix(values);
                matrices.add(realMatrix);
            }
            previousLayerSize = layerSize;
        }
        return matrices;
    }

    private double[][] initArray(int rows, int columns) {
        double[][] values = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                values[i][j] = weightSupplier.get();
            }
        }
        return values;
    }
}
