classdef SelectLayer < handle
    properties
        init = Data();
        index
        output
        input
    end
    methods
        function obj = SelectLayer(option)
            if nargin >= 1
                obj.init = Data(option);
            end
        end
        
        function output = fprop(obj,input,index)
            if nargin <= 2
                index = size(input,2);
            end
            obj.index = index;
            obj.output = cell([2,size(index,2)]);
            obj.input = input;
            obj.init.setDataSize(size(input{1,1}));
            obj.init.setZeros();
            for i = 1 : size(index,2)
                obj.output{1,i} = input{1,index(1,i)};
                obj.output{2,i} = input{2,index(1,i)};
            end
            output = obj.output;
        end
        
        function grad_input = bprop(obj,grad_output)
            obj.grad_input = cell([1,size(obj.input,2)]);
            for i = 1 : size(obj.grad_input,2)
                for j = 1 : size(obj.index,2)
                    if i ~= obj.index(1,j)
                        obj.grad_input{1,i} = obj.init.context;
                    else
                        obj.grad_input{1,i} = grad_output{1,j};
                    end
                end
            end
            grad_input = obj.grad_input;
        end
        
%         function update(obj,apply)
%         end
    end
end