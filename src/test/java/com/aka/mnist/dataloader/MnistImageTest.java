package com.aka.mnist.dataloader;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
        };
        byte label = 1;
        image = new MnistImage(label, data);
    }

    @Test
    void getDataAsList() {
        List<Integer> expectedList = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8);
        List<Integer> actualList = image.getDataAsList();
        assertEquals(expectedList, actualList);
    }
}
