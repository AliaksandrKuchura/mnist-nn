package com.aka.mnist.dataloader;

import java.util.List;

/**
 * Created by Aliaksandr Kuchura on Aug, 2020
 */

public interface Loader {

    List<MnistImage> loadFromFiles(String imagesFilePath, String labelsFilePath);

}
