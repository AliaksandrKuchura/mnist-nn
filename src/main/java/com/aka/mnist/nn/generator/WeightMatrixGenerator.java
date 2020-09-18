package com.aka.mnist.nn.generator;

import com.aka.mnist.config.Layer;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.List;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

public interface WeightMatrixGenerator {

    List<RealMatrix> createListFromConfig(List<Layer> layerConfigs);
}
