classdef DataLayer < handle
    properties 
        data = [];
    end
    properties (SetAccess = private,GetAccess = public)
        dataSize = [];
        rangeData = 0.01;
        useGPU = false;
        dataType = 'double';
        debug = false;
    end
    
    methods
        function obj = DataLayer(option)
            if nargin >= 1
                obj.initialOption(option);
                if isfield(option,'globalOption')
                    obj.initialOption(option.globalOption);
                end
            end
        end
        
        function initialOption(obj,option)
            if isfield(option,'useGPU')
                obj.setGPU(option.useGPU);
            end

            if isfield(option,'dataType')
                obj.setDataType(option.dataType);
            end

            if isfield(option,'dataSize')
                obj.setDataSize(option.dataSize);
            end

            if isfield(option,'rangeData')
                obj.setRange(option.rangeData);
            end
            
            if isfield(option,'debug')
                obj.setDebug(option.debug);
            end
        end
        
        function data = dataConvert(obj,data)
            if obj.debug
                if nargin <= 1
                    error('you should input something to convert!');
                end
            end
            if obj.useGPU
                if ~isa(class(data),'gpuArray')
                    data = gpuArray(data);
                end
            else
                if ~isa(class(data),obj.dataType)
                    eval(['data = ',obj.dataType,'(data);']);
                end
            end
        end
        
        function setData(obj,data)
            if obj.debug
                if nargin <= 1
                    error('you should input something to set the Data!');
                end
                if obj.useGPU
                    if ~isa(data,'gpuArray')
                        error('you must ensure the input data has the same dataType');
                    end
                else
                    if isa(data,'gpuArray')
                        error('you must ensure the input data has the same dataType');
                    end
                end
            end
            obj.data = data;
        end
        
        function initialData(obj)
            if obj.debug
                if isempty(obj.dataSize)
                    error('you should define the dataSize first');
                end
            end
            
            if obj.useGPU
                obj.data = 2 * obj.rangeData * (rand(obj.dataSize,obj.dataType, 'gpuArray') - 0.5);
            else
                obj.data = 2 * obj.rangeData * (rand(obj.dataSize,obj.dataType) - 0.5);
            end
        end
        
        function setZeros(obj)
            if obj.debug
                if isempty(obj.dataSize)
                    error('you should define the dataSize first');
                end
            end
            
            if obj.useGPU
                obj.data = zeros(obj.dataSize,obj.dataType, 'gpuArray');
            else
                obj.data = zeros(obj.dataSize,obj.dataType);
            end
        end
        
        function setDataSize(obj,dataSize)
            if isempty(obj.data)
                obj.dataSize = dataSize;
            else
                if obj.debug
                    if sum(size(obj.data) ~= dataSize) ~= 0
                        error('data already exist,unable to change its size');
                    end
                end
            end
        end
        
        function dataSize = getDataSize(obj)
            obj.dataSize = size(obj.data);
            dataSize = obj.dataSize;
        end
        
        function setRange(obj,rangeData)
            obj.rangeData = rangeData;
        end
        
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
        
        function setDebug(obj,debug)
            if debug == true || debug == false
                obj.debug = debug;
            else
                error('unkown debug type');
            end
        end
    end
end