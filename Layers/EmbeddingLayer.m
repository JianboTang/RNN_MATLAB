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
            obj.batch_size = size(input{1,1},2);
            for i = 1 : obj.length
                obj.input{1,i} = input{1,i};
                obj.input{2,i} = input{2,i};
                obj.output{1,i} = obj.W.context(:,obj.input{1,i});
                obj.output{2,i} = input{2,i};% mask
            end
            output = obj.output;
            if obj.debug
                display(['EmbeddingLayer | W | mean : ',num2str(mean(obj.W.context(:))),' | std : ',num2str(std(obj.W.context(:)))]);
            end
        end
        
        function output = fprop_step(obj,input,i)
            obj.length = i;
            obj.batch_size = size(input{1,1},2);
            obj.input{1,i} = input{1,1};
            obj.input{2,i} = input{2,1};
            obj.output{1,i} = obj.W.context(:,input{1,1});
            obj.output{2,i} = obj.input{2,1};% mask
            output{1,1} = obj.output{1,i};
            output{2,1} = obj.output{2,i};
        end
        
        function bprop(obj,grad_output,length)
            if nargin >= 3
                obj.length = length;
            end
            for i = 1 : obj.length
                temp = grad_output{1,i};
                temp_mask = obj.input{2,i};
                index = obj.input{1,i};
                for j = 1 : obj.batch_size
                    if temp_mask(1,j) == 1
                        obj.grad_W.context(:,index(1,j)) = obj.grad_W.context(:,index(1,j)) + temp(:,j) ./ obj.batch_size;
                    end
                end
            end
            if obj.debug
                display(['EmbeddingLayer | grad_W | mean : ',num2str(mean(obj.grad_W.context(:))),' | std : ',num2str(std(obj.grad_W.context(:)))]);
            end
        end
        
        function update(obj,apply,option)
            if nargin <= 2
                option = struct();
            end
            obj.W.context = apply(obj.W.context,obj.grad_W.context,option);
            obj.grad_W.setZeros();
        end
        
        function object = saveObj(obj)
            if obj.W.useGPU
                object.W = gather(obj.W.context);
                object.B = gather(obj.B.context);
            else
                object.W = obj.W.context;
                object.B = obj.B.context;
            end
            
            object.activation = obj.activation;
            object.diff_activ = obj.diff_activ;
        end
        
        function loadObj(obj,object)
            obj.W.context = obj.init.dataConvert(object.W);
            obj.B.context = obj.init.dataConvert(object.B);
            obj.activation = object.activaiton;
            obj.diff_activ = object.diff_activ;
        end
    end
end