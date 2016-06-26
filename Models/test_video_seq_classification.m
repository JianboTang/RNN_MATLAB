clc;
clear;
close all;
setup;
addpath(genpath('./'))
% dbstop if error
load video_sample
%% data Convert  
useGPU = false;
dataType = 'single';
backward = true;
% debug = false;

option = struct('useGPU',useGPU,'dataType',dataType);
convert = Data(option);

temp = traindata{1,1};
temp_label = trainlabel{1,1};
TrainData = cell([1,size(temp,2)]);
TrainLabel = cell([1,size(temp_label,2)]);
for i = 1  : size(temp,2)
    % ?????
    TrainData{1,i} = convert.dataConvert(temp(:,i));
end
TrainLabel{1,1} = convert.dataConvert(trainlabel{1,1});

%% ???
rec1 = LstmLayer(struct('hidden_dim',300,'input_dim',600,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug));
rec2 = LstmLayer(struct('hidden_dim',50,'input_dim',300,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug));
loss = SoftmaxLayer(struct('hidden_dim',7,'input_dim',50,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug));

MaxIter = 10000;
history_cost = zeros(1,MaxIter);
lstm_output = cell([1,1]);
for i = 1 : MaxIter
    tic;
    % ????
    rec1.fprop(TrainData,size(TrainData,2));
    rec2.fprop(rec1.output,size(TrainData,2));
    lstm_output{1,1} = rec2.output{1,size(TrainData,2)};
    loss.fprop(lstm_output,size(lstm_output,2));
    history_cost(1,i) = gather(loss.getCost(TrainLabel));
    display(' == == == == == == == == == == == == == == == ');
    display(['iteration : ',num2str(i)]);
    display(['cost      : ',num2str(history_cost(1,i))]);
    loss.bprop(TrainLabel);%????
    grad_output = cell([1,size(TrainData,2)]);
    convert.setDataSize(size(loss.grad_input{1,1}))
    convert.setZeros();
    for j = 1 : size(TrainData,2)
        grad_output{1,j} = convert.context;
    end
    grad_output{1,size(TrainData,2)} = loss.grad_input{1,1};
    rec1.bprop(rec2.bprop(grad_output));
    loss.update(@SGD);
    rec2.update(@SGD);
    rec1.update(@SGD);
    toc;
end





