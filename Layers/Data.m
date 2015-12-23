classdef Data < handle
    properties
        context
        dataSize
        rangeData = 0.1;
        useGPU = false;
        dataType = 'double';
        seed = 123 
        debug = false;
    end
    
    methods
        function obj = Data(option)
            if nargin >= 1
                obj.initialOption(option);
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
            
            if isfield(option,'seed')
                obj.setSeed(option.seed);
            end
            
            if isfield(option,'globalOpt')
                obj.initialOption(option.globalOpt);
            end
        end
        
        function context = dataConvert(obj,context)
            if obj.debug
                if nargin <= 1
                    error('you should input something to convert!');
                end
            end
            if obj.useGPU
                if ~isa(class(context),'gpuArray')
                    context = gpuArray(context);
                end
            else
                if ~isa(class(context),obj.dataType)
                    eval(['context = ',obj.dataType,'(context);']);
                end
            end
        end
        
        function setData(obj,context)
            if obj.debug
                if nargin <= 1
                    error('you should input something to set the Data!');
                end
                if obj.useGPU
                    if ~isa(context,'gpuArray')
                        error('you must ensure the input context has the same dataType');
                    end
                else
                    if isa(context,'gpuArray')
                        error('you must ensure the input context has the same dataType');
                    end
                end
            end
            obj.context = context;
        end
        
        function initial(obj)
            if obj.debug
                if isempty(obj.dataSize)
                    error('you should define the dataSize first');
                end
            end
            rng(obj.seed);
            if obj.useGPU
                obj.setData(2 * obj.rangeData * (rand(obj.dataSize,'gpuArray') - 0.5));
            else
                obj.setData(2 * obj.rangeData * (rand(obj.dataSize,obj.dataType) - 0.5));
            end
        end
        
        function setZeros(obj)
            if obj.debug
                if isempty(obj.dataSize)
                    error('you should define the dataSize first');
                end
            end
            
            if obj.useGPU
                obj.setData(zeros(obj.dataSize, 'gpuArray'));
            else
                obj.setData(zeros(obj.dataSize,obj.dataType));
            end
        end
        
        function setOnes(obj)
            if obj.debug
                if isempty(obj.dataSize)
                    error('you should define the dataSize first');
                end
            end
            
            if obj.useGPU
                obj.setData(ones(obj.dataSize,'gpuArray'));
            else
                obj.setData(ones(obj.dataSize,obj.dataType));
            end
        end
        
        function clearData(obj)
            if obj.debug
                if isempty(obj.context)
                    error('cannot clear because it is empty'); 
                end
            end
            obj.context = [];
            obj.dataSize = [];
        end
        
        function setDataSize(obj,dataSize)
            if isempty(obj.context)
                obj.dataSize = dataSize;
            else
                if sum(size(obj.context) ~= dataSize) ~= 0
                    error('context already exist,unable to change its size');
                end
            end
        end
        
        function dataSize = getDataSize(obj)
            obj.dataSize = size(obj.context);
            dataSize = obj.dataSize;
        end
        
        function setRange(obj,rangeData)
            if isnumeric(rangeData)
                obj.rangeData = rangeData;
            else
                error('unknown rangeData type,must be a numeric');
            end
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
        
        function setSeed(obj,seed)
            if isnumeric(seed)
                obj.seed = seed;
            else
                error('unknown seed type,must be a numeric');
            end
        end
    end
end