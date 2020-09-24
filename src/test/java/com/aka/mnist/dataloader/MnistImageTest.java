package com.aka.mnist.dataloader;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

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
        image = new MnistImage((byte) 1, data);
    }

    @Test
    void getDataAsVector() {
        int[] expectedList = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        int[] actualList = image.getDataAsVector();
        assertArrayEquals(expectedList, actualList);
    }

    @Test
    void testEquals() {
        int[][] data = {
                {0, 1, 2},
                {3, 4, 5},
                {6, 7, 8},
                {9, 10, 11},
        };
        MnistImage image2 = new MnistImage((byte) 1, data);

        assertEquals(image, image2);
    }

    @Test
    void testNonEquals() {
        int[][] data = {
                {0, 1, 2},
                {3, 4, 5}
        };
        MnistImage image2 = new MnistImage((byte) 1, data);

        assertNotEquals(image, image2);
    }

    @Test
    void testHashCode() {
        int expectedHashCode = 922082983;
        int actualHashcode = image.hashCode();

        assertEquals(expectedHashCode, actualHashcode);
    }

    @Test
    void testToString() {
        String expectedString = "MnistImage{label=1, data=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]}";
        String actualString = image.toString();

        assertEquals(expectedString, actualString);
    }
}
