package com.aka.mnist.config;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

@Data
@Configuration
@EnableConfigurationProperties
@ConfigurationProperties(prefix = "perceptron")
public class PerceptronLayersConfiguration {

    private String learningRate;

    private List<Layer> layers = new ArrayList<>();

    public int countLayers(){
        return layers.size();
    }


    @Data
    @Component
    @ConfigurationProperties(prefix = "perceptron.layers")
    public static class Layer {

        private String name;

        private Integer size;
    }
}
