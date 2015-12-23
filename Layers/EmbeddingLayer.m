classdef EmbeddingLayer < OperateLayer
    methods
        function obj = EmbeddingLayer(option)
            if nargin == 0
                super_args{1} = struct();
            else if nargin == 1
                    super_args{1} = option;
                end
            end
            obj = obj@OperateLayer(super_args{:});
            obj.initialOption(super_args{:});
            obj.initial();
        end
        
        function [output,length] = fprop(obj,input,length)
            obj.length = length;
            for i = 1 : obj.length
                obj.input{1,i} = input{1,i};
                obj.input{2,i} = input{2,i};
                obj.output{1,i} = obj.W.context(:,input{1,i});
                obj.output{2,i} = input{2,i};% mask
            end
            output = obj.output;
        end
        
        function output = fprop_step(obj,input,i)
            obj.length = i;
            obj.input{1,i} = input{1,1};
            obj.input{2,i} = input{2,1};
            obj.output{1,i} = obj.W.context(:,input{1,1});
            obj.output{2,i} = obj.input{2,1};% mask
            output{1,1} = obj.output{1,i};
            output{2,1} = obj.output{2,i};
        end
        
        function bprop(obj,grad_output)
            obj.grad_input = grad_output;
        end
        
        function update(obj,apply,option)
        end
    end
end