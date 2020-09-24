package com.aka.mnist.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Getter
@Setter
@Configuration
@EnableConfigurationProperties
@ConfigurationProperties(prefix = "perceptron")
public class PerceptronConfiguration {

    private double learningRate;

    private List<Layer> layers = new ArrayList<>();

    public int countLayers() {
        return layers.size();
    }

}
