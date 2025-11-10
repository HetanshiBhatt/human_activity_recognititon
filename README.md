# human_activity_recognititon
This repository contains the implementation and evaluation of a punch detection system built using Human Activity Recognition techniques. The project compares four deep learning approaches that span pose-based pipelines, CNN-RNN hybrids, and advanced spatiotemporal architectures.

Project Overview

Punch detection is a challenging HAR task due to fast motion, occlusions, camera variability, and similarity to other upper-body actions. This project investigates the trade-offs between accuracy and computational efficiency, aiming to identify architectures suitable for real-time deployment as well as high-accuracy offline analytics.

The system evaluates the following approaches:

MediaPipe + LSTM (pose-based)

MobileNetV2 + LSTM (CNN feature extractor + temporal modeling)

VGG19 + ConvLSTM (spatiotemporal convolutional architecture)

SlowFast ResNet-50 (state-of-the-art video-level spatiotemporal model)

Dataset

A binary punch vs. non-punch dataset was created using videos extracted from UCF-101.

140 punch videos

187 non-punch videos

Train/Val/Test split: 70/15/15

All videos standardized to AVI format

Temporal sampling and padding applied per model requirement

This dataset includes classes such as Punch, Applying Lipstick, Archery, and other non-punch actions to create a realistic negative class.

Preprocessing Pipeline

Different models required different preprocessing strategies:

MediaPipe + LSTM

Landmark extraction using MediaPipe Holistic (pose + hand landmarks)

30-frame sequences per sample

225-dimensional feature vector per frame

Sliding window with zero-padding for missing landmarks

MobileNetV2 + LSTM

Frame-wise preprocessing using MobileNetV2's ImageNet-normalization

224x224 RGB frames

Fixed sequence length (30 frames)

SlowFast ResNet-50

Two pathways: slow (low fps) and fast (high fps)

32-frame input clips

Short-side scaling, center cropping, and normalization

Frame-level and video-level inference pipelines

VGG19 + ConvLSTM

TimeDistributed VGG19 backbone for spatial feature extraction

20 temporally sampled frames per video

Overlapping sliding windows (stride=5)

ConvLSTM with attention mechanism

Model Architectures
MediaPipe + LSTM

Input: (30, 225)

Two LSTM layers (64 and 128 units)

Dense layer + softmax classifier

Lightweight and real-time capable

MobileNetV2 + LSTM

TimeDistributed MobileNetV2 (frozen)

GlobalAveragePooling

Two-layer LSTM for temporal context

Dense + dropout + softmax output

SlowFast ResNet-50

Dual-pathway spatiotemporal CNN

Pretrained on Kinetics-400

Fused slow semantic and fast motion streams

Highest accuracy, computationally heavy

VGG19 + ConvLSTM

Pretrained VGG19 feature extractor

ConvLSTM2D + attention

Strong theoretical design but overfitting observed

Results

Evaluation metrics include Accuracy, Precision, Recall, F1 Score, and ROC-AUC. Key findings:

SlowFast ResNet-50 (video-level) achieved near-perfect classification (Accuracy: 99.7 percent, Precision: 1.0, Recall: 0.993)

MobileNetV2 + LSTM achieved high accuracy (98 percent) with efficient inference

MediaPipe + LSTM performed competitively (95 percent accuracy) with excellent real-time performance

VGG19 + ConvLSTM underperformed due to overfitting and limited dataset generalization

SlowFast frame-level evaluation struggled due to limited temporal context

Discussion

The study highlights the trade-offs between accuracy and computational cost:

Pose-based and lightweight CNN-LSTM models are best suited for embedded or resource-constrained real-time systems.

SlowFast architectures deliver superior accuracy but require significant computation, best used for offline analytics.

Temporal modeling and video-level aggregation significantly improve recognition performance compared to frame-level approaches.
