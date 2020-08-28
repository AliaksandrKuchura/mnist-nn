package com.aka.mnist.dataloader;

import lombok.Getter;
import lombok.Setter;

import java.util.*;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Getter
@Setter
public class MnistImage {

    private byte label;

    private int[][] data;

    public MnistImage(byte label, int[][] data) {
        this.label = label;
        this.data = data;
    }

    public List<Integer> getDataAsList() {
        int rowSize = data.length;

        if (rowSize > 0) {
            List<Integer> result = new ArrayList<>();

            for (int i = 0; i < rowSize; i++) {
                int columnSize = data[i].length;

                for (int j = 0; j < columnSize; j++) {
                    result.add(data[i][j]);
                }
            }
            return result;
        } else {
            return Collections.emptyList();
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
                ", data=" + Arrays.toString(data) +
                '}';
    }
}
