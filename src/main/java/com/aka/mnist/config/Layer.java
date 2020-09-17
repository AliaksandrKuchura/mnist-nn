package com.aka.mnist.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

@Data
@Component
@ConfigurationProperties(prefix = "perceptron.layers")
public class Layer {

    private String name;

    private Integer size;
}
