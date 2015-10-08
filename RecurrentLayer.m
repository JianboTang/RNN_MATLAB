classdef RecurrentLayer < OperateLayer
    properties
        W_T = DataLayer();
        grad_W_T = DataLayer();
        last_output = DataLayer();
        grad_last_output = DataLayer();
    end
    properties (SetAccess = private,GetAccess = public,Hidden)
        activation = @tanh;
        diff_activ = @tanhPrime
    end
    methods
        function obj = RecurrentLayer(option)
            if nargin == 0
                super_args{1} = struct();
            else if nargin == 1
                    super_args{1} = option;
                end
            end
            obj = obj@OperateLayer(super_args{:});
            obj.initialOption(super_args{:});
            obj.W_T = DataLayer(super_args{:});
            obj.grad_W_T = DataLayer(super_args{:});
            obj.last_output = DataLayer(super_args{:});
            obj.grad_last_output = DataLayer(super_args{:});
        end
        
        function initial(obj)
            if obj.debug
                if isempty(obj.dimension)
                    error('you should initialize the dimension first!');
                end
            end
            initial@OperateLayer(obj);
            
            obj.W_T.setDataSize([obj.dimension(1,1),obj.dimension(1,1)]);
            obj.W_T.initialData();
%             obj.outputData.setZeros();
        end
        
        function setLastOutput(obj,data)
            obj.last_output.setData(data);
        end
        
        function forward(obj)
            if obj.debug
                if isempty(obj.W.data) || isempty(obj.W_T.data) || isempty(obj.last_output.data) || isempty(obj.inputData.data)
                    error('not all the data are initialized yet,and can not do the forward operation!');
                end
            end
            obj.outputData.setData(obj.activation((obj.W_T.data * obj.last_output.data + obj.W.data * obj.inputData.data)));
%             obj.last_output.setData(obj.outputData.data);
        end
        
        function getGrad(obj)
            % before the this operation ,you should specify the responding
            % data,for example,the grad_output,the last_output and the
            % input data.
            if obj.debug
                if isempty(obj.grad_output.data)
                    error('you should initialize the grad_output data first!');
                end
                
                if isempty(obj.last_output.data)
                    error('you should initialize the last_output data first!');
                end
                
                if isempty(obj.inputData.data)
                    error('you should intialize the intputData first!');
                end
            end
            
            grad_a = obj.grad_output.data .* obj.diff_activ(obj.outputData.data);
            obj.grad_W_T.setData(grad_a * obj.last_output.data');
            obj.grad_W.setData(grad_a * obj.inputData.data');
            obj.grad_last_output.setData(obj.W_T.data' * grad_a);
            obj.grad_input.setData(obj.W.data' * grad_a);
        end
        %% the functions below this line are used in the above functions or some functions are just defined for the gradient check;
        
        function initialOption(obj,option)
            initialOption@OperateLayer(obj,option);
            if isfield(option,'activation')
                obj.setActivation(option.activation);
            end
            
            if isfield(option,'diff_avtiv')
                obj.setDiffActiv(option.diff_avtiv);
            end
        end
        
        function setActivation(obj,activation)
            % to do the check in the future,and the similar operation should
            % do in the similar functions
            obj.activation = activation;
        end
        
        function setDiffActiv(obj,diff_activ)
            obj.diff_activ = diff_activ;
        end
        
        
        
        function checkGrad(obj)
            if obj.debug
                if isfield(obj.globalOption,'useGPU')
                    if obj.globalOption.useGPU == 1
                        error('you should not use GPU in the process of check!');
                    end
                end
                if isfield(obj.globalOption,'dataType')
                    if ~strcmpi(obj.globalOption.dataType,'double')
                        error('you should  use the dataType DOUBLE!');
                    end
                end
            end
            
            num1 = 100;
            num2 = 100;
            samples = 2;
            epsilon = 10^(-8);
            obj.setDimension([num1,num2]);
            obj.initial();
            
            W = obj.W.data;
            W_T = obj.W_T.data;
            input = rand([num2,samples]);
            last_output = rand([num1,samples]);
            target = rand(size(last_output));
            obj.setInputData(input);
            obj.setLastOutput(last_output);
            obj.forward();
            
            cost = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
            obj.setGradOutput(obj.outputData.data - target);
            obj.getGrad();
            
            grad_W = obj.grad_W.data;
            grad_W_T = obj.grad_W_T.data;
            grad_last_output = obj.grad_last_output.data;
            grad_input = obj.grad_input.data;
            
            check_W = zeros(size(grad_W));
            check_W_T = zeros(size(grad_W_T));
            check_last_output = zeros(size(grad_last_output));
            check_input = zeros(size(grad_input));
            
            for i = 1 : num1
                for j = 1 : num2
                    obj.W.data = W;
                    obj.W.data(i,j) = obj.W.data(i,j) + epsilon;%use this formula as the standard numeric difference formula
                    obj.forward();
                    cost_1 = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
                    
                    obj.W.data(i,j) = W(i,j) - epsilon;
                    obj.forward();
                    cost_2 = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
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
            
            for i = 1 : num1
                for j = 1 : num1
                    obj.W_T.data(i,j) = W_T(i,j) + epsilon;
                    obj.forward();
                    cost_1 = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
                    
                    obj.W_T.data(i,j) = W_T(i,j) - epsilon;
                    obj.forward();
                    cost_2 = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
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
            
            for i = 1 : num1
                for j = 1 : samples
                    obj.last_output.data(i,j) = last_output(i,j) + epsilon;
                    obj.forward();
                    cost_1 = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
                    
                    obj.last_output.data(i,j) = last_output(i,j) - epsilon;
                    obj.forward();
                    cost_2 = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
                    check_last_output(i,j) = (cost_1 - cost_2) / (2 * epsilon);
                end
            end
            
            norm_diff = norm(grad_last_output(:) - check_last_output(:)) ./ norm(grad_last_output(:) + check_last_output(:));
            if obj.debug
                disp('check the last_output grad');
                disp([grad_last_output(:),check_last_output(:)]);
            end
            disp('the differece between these two grads is : ');
            disp(norm_diff);
            
            for i = 1 : num2
                for j = 1 : samples
                    obj.inputData.data(i,j) = input(i,j) + epsilon;
                    obj.forward();
                    cost_1 = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
                    
                    obj.inputData.data(i,j) = input(i,j) - epsilon;
                    obj.forward();
                    cost_2 = 0.5 * sum((obj.outputData.data(:) - target(:)) .^ 2);
                    check_input(i,j) = (cost_1 - cost_2) / (2 * epsilon);
                end
            end
            
            norm_diff = norm(grad_input(:) - check_input(:)) ./ norm(grad_input(:) + check_input(:));
            if obj.debug
                disp('check the last_output grad');
                disp([grad_input(:),check_input(:)]);
            end
            disp('the differece between these two grads is : ');
            disp(norm_diff);
        end
    end
end