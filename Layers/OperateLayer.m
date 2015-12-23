classdef OperateLayer < handle
    properties
        W = Data();
        B = Data();
        grad_W = Data();
        grad_B = Data();
        init = Data();
        activation = @tanh;
        diff_activ = @tanhPrime;
        length
        batch_size
        input
        output
        grad_input
        grad_output
    end
    properties(SetAccess = private,GetAccess = public)
        hidden_dim
        input_dim
        mask = true;
        debug = false;
        backward = false;
    end
    methods
        function obj = OperateLayer(option)
            if nargin >= 1
                obj.W = Data(option);
                obj.B = Data(option);
                obj.grad_W = Data(option);
                obj.grad_B = Data(option);
                obj.init = Data(option);
            end
        end
        
%         function input_initialize(obj)
%             obj.input = cell([2,obj.length]);
%             obj.output = cell([2,obj.length]);
%         end
        
%         function grad_initialize(obj)
%             obj.grad_input = cell([1,obj.length]);
%             obj.grad_output = cell([1,obj.length]);
%         end
        
        function update(obj,apply,option)
            if nargin <= 2
                option = struct();
            end
            obj.W.context = apply(obj.W.context,obj.grad_W.context,option);
            obj.B.context = apply(obj.B.context,obj.grad_B.context,option);
            obj.grad_W.setZeros();
            obj.grad_B.setZeros();
        end
        
        function initial(obj)
            if obj.debug
                if isempty(obj.hidden_dim) || isempty(obj.input_dim)
                    error('the dimension should be initialized first');
                end
            end
            
            obj.W.setDataSize([obj.hidden_dim,obj.input_dim]);
            obj.B.setDataSize([obj.hidden_dim,1]);
            obj.W.initial();
            obj.B.initial();
            if obj.backward
                obj.grad_W.setDataSize([obj.hidden_dim,obj.input_dim]);
                obj.grad_B.setDataSize([obj.hidden_dim,1]);
                obj.grad_W.setZeros();
                obj.grad_B.setZeros();
            end
        end
        
        function initialOption(obj,option)
            if isfield(option,'hidden_dim')
                obj.hidden_dim = option.hidden_dim;
            end
            
            if isfield(option,'input_dim')
                obj.input_dim = option.input_dim;
            end
            
            if isfield(option,'debug')
                obj.setDebug(option.debug);
            end
            
            if isfield(option,'activation')
                obj.activation = option.activation;
            end
            
            if isfield(option,'diff_avtiv')
                obj.diff_activ = option.diff_activ;
            end
            
            if isfield(option,'backward')
                obj.backward = option.backward;
            end
        end
        
        function setDebug(obj,debug)
            if debug == true || debug == false
                obj.debug = debug;
            else
                error('unkown debug type');
            end
        end
    end
end