clc;
clear;
close all;
setup;
dbstop if error
%%  LSTM  autoencoder

% layers definition
useGPU = true;
dataType = 'single';
backward = true;
debug = true;

input = InputLayer(struct('batchSize',5,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug));
embedd1 = EmbeddingLayer(struct('hidden_dim',512,'input_dim',input.vocabSize,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug));
rec1 = LstmLayer(struct('hidden_dim',512,'input_dim',512,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug));
rec2 = LstmLayer(struct('hidden_dim',512,'input_dim',512,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug));
loss = SoftmaxLayer(struct('hidden_dim',input.vocabSize,'input_dim',512,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug));

% train
MaxIter = 10000;
history_cost = zeros(1,MaxIter);
for i = 1 : MaxIter
    tic;
    target = input.fprop(struct('reverse',false,'fprop',true));
    loss.fprop(rec2.fprop(rec1.fprop(embedd1.fprop(target,size(target,2)),size(target,2)),size(target,2)),size(target,2));
    history_cost(1,i) = gather(loss.getCost(target));
    display(['cost of this ',num2str(i),'th iteration is ',num2str(history_cost(1,i))]);
    
    embedd1.bprop(rec1.bprop(rec2.bprop(loss.bprop(target))));
    loss.update(@SGD);
    rec2.update(@SGD);
    rec1.update(@SGD);
    embedd1.update(@SGD);
    toc;
end