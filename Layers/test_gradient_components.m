clc;
clear;
close all;
dbstop if error

%% check the gradient of RecurrentLayer
display('check the gradient of RecurrentLayer')
input_dim = 8;
hidden_dim = 5;
option = struct('hidden_dim',hidden_dim,'input_dim',input_dim,'useGPU',false,'dataType','double','backward',true);
check = RecurrentLayer(option);
check.checkGrad();

%% check the gradient of LstmLayer
display('check the gradient of LstmLayer')
input_dim = 8;
hidden_dim = 5;
option = struct('hidden_dim',hidden_dim,'input_dim',input_dim,'useGPU',false,'dataType','double','backward',true);
check = LstmLayer(option);
check.checkGrad();

%% check the gradient of SoftmaxLayer
display('check the gradient of SoftmaxLayer')
input_dim = 8;
hidden_dim = 5;
option = struct('hidden_dim',hidden_dim,'input_dim',input_dim,'useGPU',false,'dataType','double','backward',true);
check = SoftmaxLayer(option);
check.checkGrad();