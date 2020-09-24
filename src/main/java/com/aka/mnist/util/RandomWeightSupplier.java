package com.aka.mnist.util;

import org.springframework.stereotype.Component;

import java.util.function.DoubleSupplier;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Component
public class RandomWeightSupplier implements DoubleSupplier {

    @Override
    public double getAsDouble() {
        return Math.random() * 2 - 1;
    }
}
