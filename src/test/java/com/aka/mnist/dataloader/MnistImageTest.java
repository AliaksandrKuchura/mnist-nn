package com.aka.mnist.dataloader;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

class MnistImageTest {

    private MnistImage image;

    @BeforeEach
    void setUp() {
        int[][] data = {
                {0, 1, 2},
                {3, 4, 5},
                {6, 7, 8},
                {9, 10, 11},
        };
        byte label = 1;
        image = new MnistImage(label, data);
    }

    @Test
    void getDataAsVector() {
        int[] expectedList = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        int[] actualList = image.getDataAsVector();
        assertArrayEquals(expectedList, actualList);
    }
}
