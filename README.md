# Project-Title

A ML model is used to predict ICU patient mortality using features extracted from the heart signals of ICU patients.
[Presentation Link](https://docs.google.com/presentation/d/1PUlbGhWOBNUF6f4VvrAzlr92MIoJXv3GRRsidV0lIU8/edit?usp=sharing)

<hr/>

## How to install and get it up and running
To be able to process 2TB patient waveform signals for this project, we need a cluster of 6 EC2 m5ad.4xlarge for feature extraction and prediction and one  EC2 m4.large for postgresql.

<hr/>

## Introduction
Early hospital mortality prediction is critical to make efﬁcient medical decisions about the severely ill patients staying in intensive care units (ICUs).
Most of mortality prediction methods need clinical records. However, some of the laboratory test results are time-consuming and and need to be processed.

In this project heart rate signals are used to predict mortality.

## Architecture
![Untitled drawing](https://user-images.githubusercontent.com/39537957/85251001-fae2db80-b40c-11ea-8c06-2f3dc033c687.jpg)

## Dataset
MIMIC III waveform Database

## Engineering challenges
Huge amount of binary data for waveform signals and missing signals in some records were part of the challenge I faced in this project. In a addition, current modules for extracting heart rate from signals are not designed for distributed systems. The module I used needs signal file and the related header file in current directory. So, I had to read signal and related header file from S3 and save it on EC2 to extract heart rate and delete it after processing. 

To overcome these challenge I divided data in 10 batches and create a thread for each one. Feature extraction and prediction was done in spark to take advantage of distribution and paralelizem.
