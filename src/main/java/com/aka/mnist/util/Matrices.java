package com.aka.mnist.util;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

public class Matrices {

    public static double[][] multiply(double[][] matrixA, double[][] matrixB) throws UnsupportedOperationException {
        if (matrixA[0].length != matrixB.length) {
            throw new UnsupportedOperationException(
                    String.format("The column count (%d)  of 'matrixA' should be equals to the row count (%d) of 'matrixB'.",
                            matrixA[0].length, matrixB.length));
        }
        double[][] resultMatrix = new double[matrixA.length][matrixB[0].length];

        for (int row = 0; row < matrixA.length; row++) {
            for (int col = 0; col < matrixB[0].length; col++) {
                resultMatrix[row][col] = multiplyCells(matrixA, matrixB, row, col);
            }
        }
        return resultMatrix;
    }

    static double multiplyCells(double[][] matrixA, double[][] matrixB, int row, int col) {
        double result = 0;

        for (int i = 0; i < matrixA[0].length; i++) {

            result += matrixA[row][i] * matrixB[i][col];
        }
        return result;
    }
}
