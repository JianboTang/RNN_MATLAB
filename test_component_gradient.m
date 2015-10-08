dbstop if error
clc;clear;close all;

disp('check the gradient computation of RecurrentLayer');
check = RecurrentLayer();
check.checkGrad();

disp('check the gradient computagtion of RecurrentNeuralNetwork');
check = RecurrentNeuralNetwork();
check.checkGrad();

disp('check the gradient computation of SoftmaxLossLayer');
check = SoftmaxLossLayer();
check.checkGrad();



