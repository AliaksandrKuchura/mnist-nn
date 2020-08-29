package com.aka.mnist.dataloader;

import com.aka.mnist.utils.ByteConverter;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

class MnistDataLoaderTest {

    private MnistDataLoader loader = new MnistDataLoader();

    @Test
    void loadFromFiles() {
        List<MnistImage> expectedMnistImages = Arrays.asList(
                new MnistImage((byte) 1, new int[][]{{0, 1, 1}, {1, 0, 0}}),
                new MnistImage((byte) 2, new int[][]{{0, 1, 0}, {1, 1, 1}})
        );
        String labelFile =
                "src/test/resources/data/labels-valid-size-2-values-1-2.idx1-ubyte";
        String imageFile =
                "src/test/resources/data/images-valid-size-2-row2-col3-values-011100-010111.idx1-ubyte.idx1-ubyte";
        List<MnistImage> actualMnistImages = loader.loadFromFiles(imageFile, labelFile);

        assertEquals(expectedMnistImages,actualMnistImages);
    }

    @Test
    void loadFromFilesEmptyList() {
        String labelFile =
                "src/test/resources/data/labels-valid-size-2-values-1-2.idx1-ubyte";
        String imageFile =
                "src/test/resources/data/images-wrong-magic-number.idx1-ubyte";
        List<MnistImage> actualMnistImages = loader.loadFromFiles(imageFile, labelFile);

        assertTrue(actualMnistImages.isEmpty());
    }

    @Test
    void loadLabels() {
        String file = "src/test/resources/data/labels-valid-size-2-values-1-2.idx1-ubyte";
        List<Byte> expectedBytes = Arrays.asList((byte) 1, (byte) 2);
        List<Byte> actualBytes =
                loader.loadLabels(file);

        assertEquals(expectedBytes, actualBytes);
    }

    @Test
    void loadLabelsFailWithUnsupportedOperationException() {
        String file = "src/test/resources/data/labels-wrong-magic-number.idx1-ubyte";
        String expectedMessage =
                String.format("File '%s' is not supported. Expected label magic number - %d. Actual - %s.",
                        file, 2049, 2);
        UnsupportedOperationException exception =
                assertThrows(UnsupportedOperationException.class, () -> loader.loadLabels(file));
        String actualMessage = exception.getMessage();

        assertEquals(expectedMessage, actualMessage);
    }

    @Test
    void loadLabelsReturnEmptyList() {
        List<Byte> result = loader.loadLabels("wrong-file-name.txt");

        assertTrue(result.isEmpty());
    }

    @Test
    void readFile() throws IOException {
        String file = "src/test/resources/data/labels-valid-size-2-values-1-2.idx1-ubyte";
        ByteBuffer buffer = loader.readFile(file);
        int actualCapacity = buffer.capacity();
        String actualBufferHex = ByteConverter.bytesToHex(buffer.array());

        assertEquals(10, actualCapacity);
        assertEquals("00000801000000020102", actualBufferHex);
    }

    @Test
    void readFileFailWithIOException() {
        String file = "file-not-exists.txt";
        String expectedMessage =
                String.format("%s (The system cannot find the file specified)", file);
        IOException exception =
                assertThrows(IOException.class, () -> loader.readFile(file));
        String actualMessage = exception.getMessage();

        assertEquals(expectedMessage, actualMessage);
    }

    @Test
    void loadImageMatrices() {
        String file = "src/test/resources/data/images-valid-size-2-row2-col3-values-011100-010111.idx1-ubyte.idx1-ubyte";
        List<int[][]> imageMatrices = loader.loadImageMatrices(file);
        int[][] expectedValue = {{0, 1, 1}, {1, 0, 0}};
        int[][] expectedValue2 = {{0, 1, 0}, {1, 1, 1}};

        assertEquals(2, imageMatrices.size());
        assertArrayEquals(expectedValue, imageMatrices.get(0));
        assertArrayEquals(expectedValue2, imageMatrices.get(1));
    }

    @Test
    void loadImageMatricesFailWithUnsupportedOperationException() {
        String file = "src/test/resources/data/images-wrong-magic-number.idx1-ubyte";
        String expectedMessage =
                String.format("File '%s' is not supported. Expected image magic number - %d. Actual - %s.",
                        file, 2051, 2048);
        UnsupportedOperationException exception =
                assertThrows(UnsupportedOperationException.class, () -> loader.loadImageMatrices(file));
        String actualMessage = exception.getMessage();

        assertEquals(expectedMessage, actualMessage);
    }

    @Test
    void convertToMnistImageList() {
        List<Byte> labelList = Arrays.asList((byte) 1, (byte) 2);
        List<int[][]> imageMatrixList = new ArrayList<>();
        int[][] imageMatrix = {{0, 1}, {2, 3}};
        int[][] imageMatrix2 = {{4, 5}, {6, 7}};
        imageMatrixList.add(imageMatrix);
        imageMatrixList.add(imageMatrix2);
        List<MnistImage> actualList = loader.convertToMnistImageList(labelList, imageMatrixList);

        assertEquals(2, actualList.size());

        for (int i = 0; i < actualList.size(); i++) {
            MnistImage mnistImage = actualList.get(i);

            assertEquals(labelList.get(i), mnistImage.getLabel());
            assertArrayEquals(imageMatrixList.get(i), mnistImage.getData());
        }
    }

    @Test
    void convertToMnistImageListFailWithUnsupportedOperationException() {
        List<Byte> labelList = Arrays.asList((byte) 1, (byte) 2);
        List<int[][]> imageMatrixList = new ArrayList<>();
        int[][] imageMatrix = {{0, 1}, {2, 3}};
        imageMatrixList.add(imageMatrix);

        String expectedMessage =
                String.format("MNIST images can't be created from presented files because list sizes aren't the same." +
                        " Label list - %d. Image matrix list - %d.", labelList.size(), imageMatrixList.size());
        UnsupportedOperationException exception =
                assertThrows(UnsupportedOperationException.class,
                        () -> loader.convertToMnistImageList(labelList, imageMatrixList));
        String actualMessage = exception.getMessage();

        assertEquals(expectedMessage, actualMessage);
    }
}
