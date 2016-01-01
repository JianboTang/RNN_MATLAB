classdef ConcatLayer < OperateLayer
    properties
        input_index
    end
    methods
        function obj = ConcatLayer(option)
            if nargin == 0
                super_args{1} = struct();
            else if nargin == 1
                    super_args{1} = option;
                end
            end
            obj = obj@OperateLayer(super_args{:});
        end
        
        function output = fprop(obj,input,length)
            obj.length = length;
            obj.input = input;
            obj.input_index = zeros([size(input,1) + 1,1]);
            obj.batch_size = size(obj.input{1,1},2);
            obj.input_index(1,1) = 1;
            for j = 1 : size(input,1)
                obj.input_index(j + 1,1) = obj.input_index(j,1) + size(input{j,1},1);
            end
            for i = 1 : obj.length
                if isempty(obj.output)
                    obj.output = cell([1,obj.length]);
                end
                if isempty(obj.output{1,i})
                    obj.init.setDataSize([obj.input_index(end,1) - 1,obj.batch_size]);
                    obj.init.setZeros();
                    obj.output{1,i} = obj.init.context;
                    obj.init.clearData();
                end
            end
            for i = 1 : obj.length
                temp = obj.output{1,i};
                for j = 1 : size(obj.input,1)
                    temp(obj.input_index(j,1) : obj.input_index(j + 1,1) - 1,: ) = obj.input{j,i};
                end
                obj.output{1,i} = temp;
            end
            output = obj.output;
        end
        
        function grad_input = bprop(obj,grad_output,length)
            if nargin >= 3
                obj.length = length;
            end
            for i = obj.length : -1 : 1
                obj.grad_output{1,i} = grad_output{1,i};
                for j = 1 : size(obj.input_index,1) - 1
                    obj.grad_input{j,i} = obj.grad_output{1,i}(obj.input_index(j,1) : obj.input_index(j + 1,1) - 1,:);
                end
            end
            grad_input = obj.grad_input;
        end
    end
end