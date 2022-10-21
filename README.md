# Fault-Detection

# Dataset
## Description:

Based on the [MAFAULDA](http://www02.smt.ufrj.br/~offshore/mfs/page_01.html) project datased available also on [ Kaggle - Machinery Fault Dataset](https://www.kaggle.com/uysalserkan/fault-induction-motor-dataset)

This database comprises on samples taken from a rate of 50 kHz scanning A/D device using the SpectraQuest Inc. Alignment/Balance Vibration Trainer (ABVT) Machinery Fault Simulator (MFS) as shown below:
![Machinery Fault Simulator](https://spectraquest.com/spectraquest/images/products/main/MFS.jpg)
Source: [Machine Fault Simulator](https://spectraquest.com/machinery-fault-simulator/details/mfs/)

For more details, reach the MAFAULDA project as mentioned on the link above.

## Dataset format:

This database is composed of 1951 multivariate time-series acquired by sensors on a SpectraQuest's Machinery Fault Simulator (MFS) Alignment-Balance-Vibration (ABVT). The 1951 comprises six different simulated states: normal function, imbalance fault, horizontal and vertical misalignment faults and, inner and outer bearing faults. This section describes the database.

The database is composed by several CSV (Comma-Separated Values) files, each one with 8 columns, one column for each sensor, according to:

* column 1 - tachometer signal that allows to estimate rotation frequency;

* columns 2 to 4 - underhang bearing accelerometer (axial, radiale tangential direction);

* columns 5 to 7 - overhang bearing accelerometer (axial, radiale tangential direction);

* column 8 - microphone.

And making a simple exploratory analysis, that is what the data extract from the columns 2 to 7 looks like:

![Dataset](https://fantinatti.com/ds/Dataset.gif)

# Models used

## ResNet
The residual neural network (ResNet) (ANN). It is a gateless or open-gated variation of the HighwayNet, which was the first functionally complete, extremely deep feedforward neural network with hundreds of layers—much deeper than earlier neural networks. To skip some levels, utilize shortcuts or skip connections (HighwayNets may also learn the skip weights themselves through an additional weight matrix for their gates). Typical ResNet models are constructed using batch normalization in between double- or triple-layer skips that contain ReLU nonlinearities.
The model architecture can be found here:
[![Resnet.png](https://i.postimg.cc/BbtJ5Ff1/Resnet.png)](https://postimg.cc/7J8vwf6x)

## FCN
Fully Convolutional Networks (FCNs) were first developed for segmentation tasks and are extremely effective at extracting features from input data. The FCN utilized for TSC is built by stacking three blocks, each of which consists of a convolutional layer with filters, followed by a batch normalization layer and a ReLU activation layer. Following the first three convolutional blocks, a global average pooling layer is applied to the features, substantially lowering the amount of weights. Finally, the softmax layer generates the final result.

## MLSTM-FCN
The Multivariate LSTM Fully Convolutional Network has produced the best results on the multivariate UEA archive. which is made up of a fully convolutional block and an LSTM block. The input multivariate time series is sent via a shuffle layer before being passed through an LSTM block with an attention mechanism followed by dropout. The output of the attention LSTM layer is concatenated with the output of the global pooling layer, and the final results are generated from the softmax layer. The inclusion of the squeeze and excite blocks provided by the authors, as well as the introduction of a feed forward link from one layer to all successive layers, is an interesting approach adopted by the authors.
## LSTM
This model was adapted to our problem and has been modified to be able to classify the 6 classes of the dataset

## Inceprion Time
The InceptionTime model is composed of five inception networks, with each prediction weighted equally, which is quite similar to the behavior of ResNet. The classifiers in the Inception network consist of two distinct residual blocks, unlike ResNet, which consists of three blocks. Each block in the inception network consists of three Inception modules rather than standard fully convolutional layers. The input of each residual block is sent to the input of the next block via a shortcut linear connection , which minimizes the vanishing gradient problem by allowing a direct flow of the gradient. The InceptionTime model is mainly used for time series classification.
