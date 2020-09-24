package com.aka.mnist.config;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Created by Aliaksandr Kuchura on Sep, 2020
 */

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Component
@ConfigurationProperties(prefix = "perceptron.layers")
public class Layer {

    private String name;

    private Integer size;
}
