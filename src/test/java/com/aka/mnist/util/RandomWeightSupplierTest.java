package com.aka.mnist.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

class RandomWeightSupplierTest {

    private RandomWeightSupplier supplier = new RandomWeightSupplier();

    @Test
    void get() {
        for (int i = 0; i < 10; i++) {
            Double value = supplier.getAsDouble();
            assertTrue(value > -1);
            assertTrue(value < 1);
        }
    }
}
