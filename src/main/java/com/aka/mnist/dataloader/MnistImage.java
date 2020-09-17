package com.aka.mnist.dataloader;

import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;
import java.util.Objects;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Getter
@Setter
public class MnistImage {

    public static final int MAX_DATA_VALUE = 255;

    private byte label;

    private int[][] data;

    MnistImage(byte label, int[][] data) {
        this.label = label;
        this.data = data;
    }

    public int[] getDataAsVector() {
        int rows = data.length;

        if (rows > 0) {
            int columns = data[1].length;
            int[] dataArray = new int[rows * columns];

            for (int i = 0; i < rows; i++) {
                System.arraycopy(data[i], 0, dataArray, i * columns, columns);
            }
            return dataArray;
        } else {
            return new int[0];
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MnistImage that = (MnistImage) o;
        return label == that.label &&
                Arrays.deepEquals(data, that.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(label);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }

    @Override
    public String toString() {
        return "MnistImage{" +
                "label=" + label +
                ", data=" + Arrays.deepToString(data) +
                '}';
    }
}
