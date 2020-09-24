package com.aka.mnist.dataloader;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Component
public class MnistDataLoader implements DataLoader {

    private static final Logger logger = LogManager.getLogger(MnistDataLoader.class);

    private static final int IMAGE_MAGIC_NUMBER = 2051;

    private static final int LABEL_MAGIC_NUMBER = 2049;

    @Override
    public List<MnistImage> loadFromFiles(String imagesFilePath, String labelsFilePath) {
        try {
            logger.info("Loading images from file '{}'...", imagesFilePath);
            List<int[][]> imageMatrices = loadImageMatrices(imagesFilePath);
            logger.info("Images from file '{}' loaded", imagesFilePath);

            logger.info("Loading labels from file '{}'...", labelsFilePath);
            List<Byte> labels = loadLabels(labelsFilePath);
            logger.info("Labels from file '{}' loaded", labelsFilePath);

            return convertToMnistImageList(labels, imageMatrices);
        } catch (UnsupportedOperationException e) {
            logger.error(e);
        }
        return Collections.emptyList();
    }

    List<Byte> loadLabels(String labelsFilePath) {
        try {
            ByteBuffer buffer = readFile(labelsFilePath);
            int magicNumber = buffer.getInt();

            if (magicNumber != LABEL_MAGIC_NUMBER) {
                throw new UnsupportedOperationException(
                        String.format("File '%s' is not supported. Expected label magic number - %d. Actual - %d.",
                                labelsFilePath, LABEL_MAGIC_NUMBER, magicNumber));
            }
            int labelsCount = buffer.getInt();
            List<Byte> labels = new ArrayList<>(labelsCount);

            while (buffer.hasRemaining()) {
                labels.add(buffer.get());
            }
            return labels;
        } catch (IOException e) {
            logger.error(e);
        }
        return Collections.emptyList();
    }

    ByteBuffer readFile(String file) throws IOException {
        try (
                RandomAccessFile randomAccessFile = new RandomAccessFile(file, "r");
                FileChannel channel = randomAccessFile.getChannel();
        ) {
            ByteBuffer buffer = ByteBuffer.allocate((int) channel.size());
            channel.read(buffer);
            buffer.flip();
            return buffer;
        }
    }

    List<int[][]> loadImageMatrices(String imagesFilePath) {
        try {
            ByteBuffer buffer = readFile(imagesFilePath);
            int magicNumber = buffer.getInt();

            if (magicNumber != IMAGE_MAGIC_NUMBER) {
                throw new UnsupportedOperationException(
                        String.format("File '%s' is not supported. Expected image magic number - %d. Actual - %d.",
                                imagesFilePath, IMAGE_MAGIC_NUMBER, magicNumber));
            }
            int labelsCount = buffer.getInt();
            int imageHeight = buffer.getInt();
            int imageWidth = buffer.getInt();

            List<int[][]> imageMatrices = new ArrayList<>(labelsCount);
            int[][] imageMatrix = new int[imageHeight][imageWidth];
            int i = 0;
            int j = 0;

            while (buffer.hasRemaining()) {
                imageMatrix[i][j] = buffer.get() & 0xFF;

                if (i == imageHeight - 1 && j == imageWidth - 1) {
                    imageMatrices.add(imageMatrix);
                    imageMatrix = new int[imageHeight][imageWidth];
                    i = 0;
                    j = 0;
                    continue;
                }
                if (j == imageWidth - 1) {
                    j = 0;
                    ++i;
                } else {
                    ++j;
                }
            }
            return imageMatrices;
        } catch (IOException e) {
            logger.error(e);
        }
        return Collections.emptyList();
    }

    List<MnistImage> convertToMnistImageList(List<Byte> labelList, List<int[][]> imageMatrixList) {
        int labelSize = labelList.size();
        int imageMatrixSize = imageMatrixList.size();

        if (labelSize != imageMatrixSize) {
            throw new UnsupportedOperationException(
                    String.format("MNIST images can't be created from presented files because list sizes aren't the same." +
                            " Label list - %d. Image matrix list - %d.", labelSize, imageMatrixSize));
        }

        List<MnistImage> result = new ArrayList<>(labelSize);
        for (int i = 0; i < labelSize; i++) {
            result.add(new MnistImage(labelList.get(i), imageMatrixList.get(i)));
        }
        return result;
    }
}
