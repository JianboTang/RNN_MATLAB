classdef RecurrentNeuralNetwork < handle
    properties
        layers = RecurrentLayer();
        grad_W = DataLayer();
        grad_W_T = DataLayer();
        inputData = DataLayer();
        grad_input = DataLayer();
        outputData = DataLayer();
        grad_output = DataLayer();
    end
    properties (SetAccess = private,GetAccess = public)
        % about the parameters of recurrent layers
        activations = {@tanh,@tanh};
        diff_activs = {@tanhPrime,@tanhPrime};
        hiddenNetSize = [20,20];
        %about the parameters of dataType 
        globalOption = struct();
        %about the parameter input data 
        batchSize = [];
        inputSize = [];
        debug = true;
        times = [];
    end
    methods
        function obj = RecurrentNeuralNetwork(option)
            if nargin >= 1
                obj.initialOption(option);
                if isfield(option,'recurrentOption')
                    obj.initialOption(option.recurrentOption);
                end
                
                if isfield(option,'globalOption')
                    obj.initialOption(option.globalOption);
                end
            end
            
            for i = 1 : size(obj.hiddenNetSize,2)
                obj.layers(1,i) = RecurrentLayer(obj.globalOption);
                obj.grad_W(1,i) = DataLayer(obj.globalOption);
                obj.grad_W_T(1,i) = DataLayer(obj.globalOption);
            end
        end
        
        function setInputData(obj,input)
            obj.setTimes(size(input,3));
            obj.setBatchSize(size(input,2));
            obj.setInputSize(size(input,1));
            
            obj.initialAdditionData();
            for i = 1 : obj.times
                obj.inputData(i,1).setData(input(:,:,i));
            end
        end
        
        function initial(obj)
            if obj.debug
                if isempty(obj.inputSize)
                    error('you should initialize the inputSize first!');
                end
            end
            
            for i = 1 : size(obj.hiddenNetSize,2)
                if i == 1
                    obj.layers(1,i).setDimension([obj.hiddenNetSize(1,i),obj.inputSize]);
                else
                    obj.layers(1,i).setDimension([obj.hiddenNetSize(1,i),obj.hiddenNetSize(1,i - 1)]);
                end
                obj.layers(1,i).setActivation(obj.activations{1,i});
                obj.layers(1,i).setDiffActiv(obj.diff_activs{1,i});
                obj.layers(1,i).initial();
            end
        end
        
        function forward(obj)
            if obj.debug
                if isempty(obj.batchSize)
                    error('you should initialize the inputData first');
                end
            end
            for j = 1 : obj.times
                for i = 1 : size(obj.layers,2)
                    if j == 1
                        obj.layers(1,i).last_output.setDataSize([obj.layers(1,i).dimension(1,1),obj.batchSize]);
                        obj.layers(1,i).last_output.setZeros();
                    else
                        obj.layers(1,i).setLastOutput(obj.outputData(j - 1,i).data);
                    end
                    if i == 1
                        obj.layers(1,i).setInputData(obj.inputData(j,1).data);
                    else
                        obj.layers(1,i).setInputData(obj.layers(1,i - 1).outputData.data);
                    end
                    obj.layers(1,i).forward();
                    obj.outputData(j,i).setData(obj.layers(1,i).outputData.data);
                end
            end
        end
        
        function getGrad(obj)
            if obj.debug
                for j = 1 : obj.times
                    if isempty(obj.grad_output(j,1).data)
                        error('you should initialize the grad_output first');
                    end
                end
            end
            
            for j = obj.times : -1 : 1
                for i = size(obj.layers,2) : -1 : 1
                    % prepare the grad_output data
                    if j == obj.times
                        if i == size(obj.layers,2)
                            obj.layers(1,i).grad_output.setData(obj.grad_output(j,1).data); 
                        else
                            obj.layers(1,i).grad_output.setData(obj.layers(1,i + 1).grad_input.data);
                        end
                    else
                        if i == size(obj.layers,2)
                            obj.layers(1,i).grad_output.setData(obj.grad_output(j,1).data + obj.layers(1,i).grad_last_output.data);
                        else
                            obj.layers(1,i).grad_output.setData(obj.layers(1,i + 1).grad_input.data + obj.layers(1,i).grad_last_output.data);
                        end
                    end
                    
                    % prepare the last output
                    if j == 1
                        obj.layers(1,i).last_output.setDataSize([obj.layers(1,i).dimension(1,1),obj.batchSize]);
                        obj.layers(1,i).last_output.setZeros();
                    else
                        obj.layers(1,i).setLastOutput(obj.outputData(j - 1,i).data);
                    end
                    
                    % prepare the outputData
                    obj.layers(1,i).outputData.setData(obj.outputData(j,i).data);
                    
                    %prepare the inputData
                    if i == 1
                        obj.layers(1,i).inputData.setData(obj.inputData(j,1).data);
                    else
                        obj.layers(1,i).inputData.setData(obj.outputData(j,i - 1).data);
                    end
                    obj.layers(1,i).getGrad();
                    if j == obj.times
                        obj.grad_W(1,i).setData(obj.layers(1,i).grad_W.data);
                        obj.grad_W_T(1,i).setData(obj.layers(1,i).grad_W_T.data);
                    else
                        obj.grad_W(1,i).setData(obj.grad_W(1,i).data + obj.layers(1,i).grad_W.data);%
                        obj.grad_W_T(1,i).setData(obj.grad_W_T(1,i).data + obj.layers(1,i).grad_W_T.data);%
                    end
                end
            end
        end
        %% the functions below this line are used in the above functions 
        
        function initialOption(obj,option)
            if isfield(option,'hiddenNetSize')
                obj.setHiddenNetSize(option.hiddenNetSize);
            end
            
            if isfield(option,'activations')
                obj.setActivations(option.activations);
            end
            
            if isfield(option,'diff_activs')
                obj.setDiffActivs(option.diff_activs);
            end
            
            if isfield(option,'globalOption')
                obj.setGlobalOption(option.globalOption);
            end
            
            if isfield(option,'batchSize')
                obj.setBatchSize(option.batchSize);
            end
            
            if isfield(option,'debug')
                obj.setDebug(option.debug);
            end
            
            if size(obj.activations,2) ~= size(obj.hiddenNetSize,2)
                error('the activations should have the same shape with the hiddenNetSize');
            end
        end
        
        function setBatchSize(obj,batchSize)
            obj.batchSize = batchSize;
        end
        
        function setDiffActivs(obj,diff_activs)
            obj.diff_activs = diff_activs;
        end
        
        function setHiddenNetSize(obj,hiddenNetSize)
            obj.hiddenNetSize = hiddenNetSize;
        end
        
        function setGlobalOption(obj,globalOption)
            obj.globalOption = globalOption;
        end
        
        function setDebug(obj,debug)
            obj.debug = debug;
        end
        
        function setInputSize(obj,inputSize)
            obj.inputSize = inputSize;
        end
        
        function setTimes(obj,times)
            obj.times = times;
        end
        
        function setActivations(obj,activations)
            obj.activations = activations;
        end
        
        function initialAdditionData(obj)
            if obj.debug
                if isempty(obj.times)
                    error('you should initialize the times first!');
                end
            end
            for j = 1 : obj.times
                for i = 1 : size(obj.layers,2)
                    obj.outputData(j,i) = DataLayer(obj.globalOption);
                end
                obj.inputData(j,1) = DataLayer(obj.globalOption);
                obj.grad_output(j,1) = DataLayer(obj.globalOption);
                obj.grad_input(j,1) = DataLayer(obj.globalOption);
            end
        end
        
        function checkGrad(obj)
            times = 20;
            batchSize = 10;
            input = rand([20,batchSize,times]);
            obj.setInputData(input);
            obj.initial();
            obj.forward();
            target = rand([20,batchSize,times]);
            cost = 0;
            for m = 1 : obj.times
                temp = target(:,:,m);
                cost = cost + 0.5 * sum((obj.outputData(m,end).data(:) - temp(:)) .^ 2);
                obj.grad_output(m,1).setData(obj.outputData(m,end).data - temp);
            end
            obj.getGrad();
            
            epsilon = 10^(-8);
            for layer = 1 : size(obj.layers,2)
                W = obj.layers(1,layer).W.data;
                grad_W = obj.grad_W(1,layer).data;
                check_W = zeros(size(W));
                for i = 1 : size(obj.layers(1,layer).W.data,1)
                    for j = 1 : size(obj.layers(1,layer).W.data,2)
                        obj.layers(1,layer).W.data = W;
                        obj.layers(1,layer).W.data(i,j) = obj.layers(1,layer).W.data(i,j) + epsilon;
                        obj.forward();

                        cost_1 = 0;
                        for m = 1 : obj.times
                            temp = target(:,:,m);
                            cost_1 = cost_1 + 0.5 * sum((obj.outputData(m,end).data(:) - temp(:)) .^ 2);
                        end

                        obj.layers(1,layer).W.data = W;
                        obj.layers(1,layer).W.data(i,j) = obj.layers(1,layer).W.data(i,j) - epsilon;
                        obj.forward();

                        cost_2 = 0;
                        for m = 1 : obj.times
                            temp = target(:,:,m);
                            cost_2 = cost_2 + 0.5 * sum((obj.outputData(m,end).data(:) - temp(:)) .^ 2);
                        end

                        check_W(i,j) = (cost_1 - cost_2) / (2 * epsilon);
                    end
                end

                norm_diff = norm(grad_W(:) - check_W(:)) ./ norm(grad_W(:) + check_W(:));
                if obj.debug
                    disp('check the weight grad');
                    disp([grad_W(:),check_W(:)]);
                end
                disp('the differece between these two grads is : ');
                disp(norm_diff);

                W_T = obj.layers(1,layer).W_T.data;
                grad_W_T = obj.grad_W_T(1,layer).data;
                check_W_T = zeros(size(W_T));
                for i = 1 : size(obj.layers(1,layer).W_T.data,1)
                    for j = 1 : size(obj.layers(1,layer).W_T.data,2)
                        obj.layers(1,layer).W_T.data = W_T;
                        obj.layers(1,layer).W_T.data(i,j) = obj.layers(1,layer).W_T.data(i,j) + epsilon;
                        obj.forward();

                        cost_1 = 0;
                        for m = 1 : obj.times
                            temp = target(:,:,m);
                            cost_1 = cost_1 + 0.5 * sum((obj.outputData(m,end).data(:) - temp(:)) .^ 2);
                        end

                        obj.layers(1,layer).W_T.data = W_T;
                        obj.layers(1,layer).W_T.data(i,j) = obj.layers(1,layer).W_T.data(i,j) - epsilon;
                        obj.forward();

                        cost_2 = 0;
                        for m = 1 : obj.times
                            temp = target(:,:,m);
                            cost_2 = cost_2 + 0.5 * sum((obj.outputData(m,end).data(:) - temp(:)) .^ 2);
                        end

                        check_W_T(i,j) = (cost_1 - cost_2) / (2 * epsilon);
                    end
                end

                norm_diff = norm(grad_W_T(:) - check_W_T(:)) ./ norm(grad_W_T(:) + check_W_T(:));
                if obj.debug
                    disp('check the weight transition grad');
                    disp([grad_W_T(:),check_W_T(:)]);
                end
                disp('the differece between these two grads is : ');
                disp(norm_diff);
            end
        end
    end
end