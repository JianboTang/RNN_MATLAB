classdef SoftmaxLossLayer < SoftmaxLayer
    properties
        target = DataLayer();
        cost = [];
    end
    methods
        function obj = SoftmaxLossLayer(option)
            if nargin == 0
                super_args{1} = struct();
            else if nargin == 1
                    super_args{1} = option;
                end
            end
            obj = obj@SoftmaxLayer(super_args{:});
            obj.grad_W = DataLayer(super_args{:});
            obj.grad_input = DataLayer(super_args{:});
            obj.target = DataLayer(super_args{:});
        end
        
        function setTarget(obj,target)
            obj.target.setData(target);
        end
        
        function getCost(obj)
            if obj.debug
                if isempty(obj.inputData.data) || isempty(obj.target.data) || isempty(obj.outputData.data)
                    error('you should initialize the input ,the net output and target first');
                end
            end
            cost_index = sub2ind(size(obj.outputData.data),obj.target.data,1 : size(obj.outputData.data,2));
            obj.cost = sum(- log(obj.outputData.data(cost_index)));
        end
        
        function getGrad(obj)
            if obj.debug
                if isempty(obj.inputData.data) || isempty(obj.target.data) || isempty(obj.outputData.data)
                    error('you should initialize the input ,the net output and target first');
                end
            end
            cost_index = sub2ind(size(obj.outputData.data),obj.target.data,1 : size(obj.outputData.data,2));
            data = obj.outputData.data;
            data(cost_index) = data(cost_index) - 1;
            obj.grad_W.setData(data * obj.inputData.data');
            obj.grad_input.setData(obj.W.data' * data);
        end
        %% the functions below this line are used in the above functions or some functions are just defined for the gradient check;
        
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
            num = 5;
            samples = 2;
            epsilon = 10^(-8);
            
            obj.inputData.setDataSize([num,samples]);
            obj.inputData.initialData();
            obj.W.setDataSize([num,num]);
            obj.W.initialData();
            obj.target.setData([1,2]);
            obj.forward();
            obj.getGrad();
            
            grad_W = obj.grad_W.data;
            grad_input = obj.grad_input.data;
            input = obj.inputData.data;
            W = obj.W.data;
            check_W = zeros(size(W));
            for i = 1 : num
                for j = 1 : num
                    obj.W.data(i,j) = W(i,j) + epsilon;
                    obj.forward();
                    obj.getCost();
                    cost_1 = obj.cost;
                    
                    obj.W.data(i,j) = W(i,j) - epsilon;
                    obj.forward();
                    obj.getCost();
                    cost_2 = obj.cost;
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
            
            check_input = zeros(size(input));
            for i = 1 : num
                for j = 1 : samples
                    obj.inputData.data(i,j) = input(i,j) + epsilon;
                    obj.forward();
                    obj.getGrad();
                    obj.getCost();
                    cost_1 = obj.cost;
                    
                    obj.inputData.data(i,j) = input(i,j) - epsilon;
                    obj.forward();
                    obj.getGrad();
                    obj.getCost();
                    cost_2 = obj.cost;
                    check_input(i,j) = (cost_1 - cost_2) / (2 * epsilon);
                end
            end
            norm_diff = norm(grad_input(:) - check_input(:)) ./ norm(grad_input(:) + check_input(:));
            if obj.debug
                disp('check the input grad');
                disp([grad_input(:),check_input(:)]);
            end
            disp('the differece between these two grads is : ');
            disp(norm_diff);
        end
    end
end