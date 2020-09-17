package com.aka.mnist.util;

import org.springframework.stereotype.Component;

import java.util.function.Supplier;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Component
public class RandomWeightSupplier implements Supplier<Double> {

    @Override
    public Double get() {
        return Math.random() * 2 - 1;
    }
}
