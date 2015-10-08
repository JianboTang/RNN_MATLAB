classdef OperateLayer < handle
    properties
        W = DataLayer();
        grad_W = DataLayer();
        inputData = DataLayer(); 
        grad_input = DataLayer();
        outputData = DataLayer();
        grad_output = DataLayer();
    end
    properties(SetAccess = private,GetAccess = public)
        dimension = []
        globalOption = struct();
        debug = true;
    end
    methods
        function obj = OperateLayer(option)
            if nargin >= 1
                obj.initialOption(option);
                
                if isfield(option,'globalOption')
                    obj.initialOption(option.globalOption);
                end
            end
            obj.W = DataLayer(obj.globalOption);
            obj.outputData = DataLayer(obj.globalOption);
            obj.inputData = DataLayer(obj.globalOption);
            obj.grad_W = DataLayer(obj.globalOption);
            obj.grad_input = DataLayer(obj.globalOption);
            obj.grad_output = DataLayer(obj.globalOption);
        end
        
        function setDimension(obj,dimension)
            if obj.debug
                if length(dimension) < 2
                    error('unsatisfied dimension,should be greater than or equal with 2!');
                end
            end
            obj.dimension = dimension;
        end
        
        function setInputData(obj,input)
            % to do the check in the future,and the similar operation should
            % do in the similar functions
            obj.inputData.setData(input);
        end
        
        function setGradOutput(obj,grad_output)
            obj.grad_output.setData(grad_output);
        end
        
        function initial(obj)
            if obj.debug
                if isempty(obj.dimension)
                    error('the dimension should be initialized first');
                end
            end
            
            obj.W.setDataSize(obj.dimension);
            obj.W.initialData();
%             obj.outputData.setDataSize([obj.dimension(1,1),1]);%size(obj.inputData.data,2)
%             obj.outputData.setZeros();
        end
        %% the functions below this line are used in the above functions 
        function initialOption(obj,option)
            if isfield(option,'dimension')
                obj.setDimension(option.dimension);
            end
            
            if isfield(option,'globalOption')
                obj.setGlobalOption(option.globalOption);
            end
            
            if isfield(option,'debug')
                obj.setDebug(option.debug);
            end
        end
        
        function setDebug(obj,debug)
            obj.debug = debug;
        end
        
        function setGlobalOption(obj,globalOption)
            obj.globalOption = globalOption;
        end
    end
end