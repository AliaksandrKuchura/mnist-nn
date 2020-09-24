package com.aka.mnist.nn;

import java.util.List;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

public interface NeuralNetwork<T1, T2> {

    T1 predict(T2 inputData);

    double train(List<T2> trainingData);
}
