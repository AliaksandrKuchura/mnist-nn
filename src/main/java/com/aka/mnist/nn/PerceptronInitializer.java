package com.aka.mnist.nn;

import com.aka.mnist.dataloader.DataLoader;
import com.aka.mnist.dataloader.MnistImage;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.event.ContextRefreshedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

@Component
public class PerceptronInitializer {

    private static final Logger logger = LogManager.getLogger(PerceptronInitializer.class);

    private static final String IMAGES_FILE = "data/train-images.idx3-ubyte";

    private static final String LABEL_FILE = "data/train-labels.idx1-ubyte";

    private NeuralNetwork<double[], MnistImage> neuralNetwork;

    private DataLoader dataLoader;

    @Autowired
    public PerceptronInitializer(NeuralNetwork<double[], MnistImage> perceptronNeuralNetwork,
                                 DataLoader mnistDataLoader) {
        this.neuralNetwork = perceptronNeuralNetwork;
        this.dataLoader = mnistDataLoader;
    }

    @EventListener(ContextRefreshedEvent.class)
    public void initialize() {
        List<MnistImage> images = dataLoader.loadFromFiles(IMAGES_FILE, LABEL_FILE);
        int batchSize = 100;
        int batches = images.size() / batchSize;

        for (int i = 0; i < 60; i++) {
            for (int j = 0; j < 1; j++) {
                int fromIndex = j * batchSize;
                int toIndex = fromIndex + batchSize;
                double accuracy = neuralNetwork.train(images.subList(fromIndex, toIndex));
                double percentageAccuracy = (1 - accuracy) * 100;
                logger.info(String.format("Epoch: %4d Batch: %4d Network accuracy: %.3f %%", i + 1, j + 1, percentageAccuracy));
            }
        }
        predict(images.get(0));
        predict(images.get(10));
        predict(images.get(11));
        predict(images.get(12));
        predict(images.get(9));
        System.lineSeparator();
    }

    private void predict(MnistImage mnistImage) {
        StringBuilder builder = new StringBuilder();
        builder.append("Expected value: ")
                .append(mnistImage.getLabel())
                .append(System.lineSeparator())
                .append("Network predict: ")
                .append(System.lineSeparator());

        double[] predict = neuralNetwork.predict(mnistImage);

        for (int i = 0; i < predict.length; i++) {
            double v = predict[i];
            builder.append(i)
                    .append(" - ")
                    .append((int) (v * 100))
                    .append("%")
                    .append(System.lineSeparator());

        }
        logger.info(builder.toString());
    }
}
