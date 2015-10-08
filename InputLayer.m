classdef InputLayer < handle
    properties
        data = textData('../Data/Text/test.txt','../Data/Text/dictionary');
        outputData = DataLayer();
    end
    
    properties(SetAccess = private,GetAccess = public)
        batchSize = 16
    end
    methods
        function obj = InputLayer(option)
            if nargin >= 1
                obj.initialOption(option);
                obj.outputData = DataLayer(option);
            end
        end
        
        function getFile(txtfile,dictionary)
            obj.data = textData(txtfile,dictionary);
        end
        
        function setBatchSize(obj,batchSize)
            % to do the check in the future,the similar operation should
            % do in the similar functions
            obj.batchSize = batchSize;
        end
        
        function forward(obj)
            rand_index = randi(size(obj.data.data,1),1,obj.batchSize);
            prune_data = obj.data.data(rand_index,:);
            mark = sum(prune_data,1) > 0;
            prune_data = prune_data(:,1 : sum(mark) + 1);
            % add the EOF(end of line) as a token
            obj.outputData.setData(obj.outputData.dataConvert(prune_data + 1));
        end
        %% the functions below this line are used in the above functions 
        
        function initialOption(obj,option)
            if isfield(option,'batchSize')
                obj.setBatchSize(option.batchSize);
            end
        end
    end
end