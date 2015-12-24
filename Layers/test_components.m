clc;
clear;
close all;
% dbstop if error
setup
useGPU = false;
dataType = 'single';
backward = true;
tic;
output1 = InputLayer(struct('batchSize',16,'useGPU',useGPU,'dataType',dataType,'backward',backward));
output2 = EmbeddingLayer(struct('hidden_dim',1024,'input_dim',output1.vocabSize,'useGPU',useGPU,'dataType',dataType,'backward',backward));
output3 = LstmLayer(struct('hidden_dim',1024,'input_dim',1024,'useGPU',useGPU,'dataType',dataType,'backward',backward));
output4 = RecurrentLayer(struct('hidden_dim',1024,'input_dim',1024,'useGPU',useGPU,'dataType',dataType,'backward',backward));
output5 = SoftmaxLayer(struct('hidden_dim',output1.vocabSize,'input_dim',1024,'useGPU',useGPU,'dataType',dataType,'backward',backward));
% output4 = RecurrentLayer(struct('hidden_dim',4096,'input_dim',4096,'useGPU',useGPU,'dataType',dataType,'backward',backward));
% output5 = RecurrentLayer(struct('hidden_dim',4096,'input_dim',4096,'useGPU',useGPU,'dataType',dataType,'backward',backward));
% output6= RecurrentLayer(struct('hidden_dim',4096,'input_dim',4096,'useGPU',useGPU,'dataType',dataType,'backward',backward));
toc;

tic;
output5.fprop(output4.fprop(output3.fprop(output2.fprop(output1.fprop()))));
toc;

target = cell(size(output1.output));
for i = 1 : size(target,2)
    target{1,i} = randi(output1.vocabSize,1,size(output1.output{1,i},2));
    target{2,i} = output1.output{2,i};
end

output5.getCost(target);