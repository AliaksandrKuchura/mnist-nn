package com.aka.mnist.util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

class MatricesTest {

    @Test
    void multiply() {
        double[][] matrixA = new double[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        double[][] matrixB = new double[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

        double[][] expectedResult = new double[][]{{50, 60}, {114, 140}, {178, 220}};
        double[][] actualResult = Matrices.multiply(matrixA, matrixB);

        assertArrayEquals(expectedResult, actualResult);
    }

    @Test
    void multiplyFailWithUnsupportedOperationException() {
        double[][] matrixA = new double[][]{{1, 2}, {3, 4}, {3, 4}};
        double[][] matrixB = new double[][]{{1, 2}};
        String expectedMessage =
                "The column count (2)  of 'matrixA' should be equals to the row count (1) of 'matrixB'.";

        UnsupportedOperationException exception =
                assertThrows(UnsupportedOperationException.class, () -> Matrices.multiply(matrixA, matrixB));
        assertEquals(expectedMessage, exception.getMessage());
    }

    @Test
    void multiplyCells() {
        double[][] matrixA = new double[][]{{1.0, 2.2, 3.4}};
        double[][] matrixB = new double[][]{{1.7}, {2.5}, {1.9}};
        double expectedValue = 13.66;

        assertEquals(expectedValue, Matrices.multiplyCells(matrixA, matrixB, 0, 0), 0.00001);
    }
}
