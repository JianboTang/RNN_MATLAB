clc;
clear;
close all;
setup;
dbstop if error
%%  RNN  autoencoder

% layers definition
useGPU = true;
dataType = 'single';
backward = true;

input = InputLayer(struct('batchSize',5,'useGPU',useGPU,'dataType',dataType,'backward',backward,'reverse',true));
embedd1 = EmbeddingLayer(struct('hidden_dim',512,'input_dim',input.vocabSize,'useGPU',useGPU,'dataType',dataType,'backward',backward));
rec1 = RecurrentLayer(struct('hidden_dim',512,'input_dim',512,'useGPU',useGPU,'dataType',dataType,'backward',backward));
rec2 = RecurrentLayer(struct('hidden_dim',512,'input_dim',512,'useGPU',useGPU,'dataType',dataType,'backward',backward));
loss = SoftmaxLayer(struct('hidden_dim',input.vocabSize,'input_dim',512,'useGPU',useGPU,'dataType',dataType,'backward',backward));

% train
MaxIter = 10000;
history_cost = zeros(1,MaxIter);
for i = 1 : MaxIter
    tic;
    target = input.fprop(struct('reverse',false,'fprop',true));
    loss.fprop(rec2.fprop(rec1.fprop(embedd1.fprop(target,size(target,2)),size(target,2)),size(target,2)),size(target,2));
    history_cost(1,i) = gather(loss.getCost(target));
    display(history_cost(1,i));
    
    embedd1.bprop(rec1.bprop(rec2.bprop(loss.bprop(target))));
    loss.update(@SGD);
    rec2.update(@SGD);
    rec1.update(@SGD);
    embedd1.update(@SGD);
    toc;
end