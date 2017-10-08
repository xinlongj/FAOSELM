close all
clear all
clc

% Example for feature reduction

TrainingData = load('TrainingDataset.txt');
IncrementData = load('IncrementDataset.txt');
TestingData = load('TestingDataset.txt');
FeatureChangeDimension = [2];
IncrementData(:,FeatureChangeDimension + 1) = []; % +1 because the first column is target
TestingData(:,FeatureChangeDimension + 1) = [];
FA_Type = 0;
Elm_Type = 1;
nHiddenNeurons = 1000;
ActivationFunction = 'sig';
Block = 200;


% Example for feature increment
%{
TrainingData = load('TrainingDataset.txt');
IncrementData = load('IncrementDataset.txt');
TestingData = load('TestingDataset.txt');
FeatureChangeDimension = [2];
TrainingData(:,FeatureChangeDimension + 1) = []; % +1 because the first column is target
FA_Type = 1;
Elm_Type = 1;
nHiddenNeurons = 1000;
ActivationFunction = 'sig';
Block = 200;
%}
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = FAOSELM(TrainingData, IncrementData, TestingData, FA_Type, FeatureChangeDimension, Elm_Type, nHiddenNeurons, ActivationFunction, Block)
% Example for feature increment