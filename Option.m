classdef Option < handle
    properties (SetAccess = private,GetAccess = public)
        useGPU = false
        dataType = 'double'
    end
    methods
        function obj = Option(option)
            if nargin >= 1
                if isfield(option,'useGPU')
                    obj.setGPU(option.useGPU);
                end
                
                if isfield(option,'dataType')
                    obj.setDataType(option.dataType);
                end
            end
        end
    end
    
    
    methods
        function setDataType(obj,dataType)
            if strcmp(dataType,'single') || strcmp(dataType,'double')
                obj.dataType = dataType;
            else
                error('unkonw dataType');
            end
        end
        
        function setGPU(obj,useGPU)
            if useGPU == true || useGPU == false
                obj.useGPU = useGPU;
            else
                error('unknown useGPU');
            end
        end
    end
end