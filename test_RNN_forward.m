dbstop if error
clc;clear;close all;
setup
tic;
option.globalOption.useGPU = true;
option.globalOption.dataType = 'single';
option.globalOption.debug = true;

option.recurrentOption.hiddenNetSize = [4000,4000,4000,4000];
option.recurrentOption.activations = {@tanh,@tanh,@tanh,@tanh};
option.recurrentOption.diff_activs = {@tanhPrime,@tanhPrime,@tanhPrime,@tanhPrime};

option.batchSize = 15;
option.phase = 'train';
option.embeddingDimension = 5000;

input = InputLayer(option);
input.forward();

wordEmbedd = EmbeddingLayer(option);
wordEmbedd.setDimension([option.embeddingDimension,size(input.data.index,1) + 1]);
wordEmbedd.setInputData(input.outputData.data);
wordEmbedd.initial();
wordEmbedd.forward();

rec = RecurrentNeuralNetwork(option);
rec.setInputData(wordEmbedd.outputData.data);
rec.initial();
rec.forward();

output = SoftmaxLossLayer(option);
output.setDimension([option.embeddingDimension,option.recurrentOption.hiddenNetSize(1,end)]);
output.setInputData(rec.outputData(1,1).data);
output.initial();
output.forward();
toc;
% layer = (option);
% end