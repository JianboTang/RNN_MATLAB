classdef InputLayer < handle
    properties
        txt = textData('../Data/Text/test_small.txt','../Data/Text/dictionary');
        output
        base_point = 1
        batchSize = 16
        mask = false
        convert = Data();
        vocabSize
        vocab
    end
    methods
        function obj = InputLayer(option)
            if nargin >= 1
                obj.initialOption(option);
                obj.convert = Data(option);
            end
            obj.vocab = obj.txt.index;
            obj.vocab{end + 1} = '<UNK>';
            obj.vocab{end + 1} = '<EOL>';
            obj.vocabSize = length(obj.vocab);
        end
        
        function getFile(txtfile,dictionary)
            obj.txt = textData(txtfile,dictionary);
            obj.vocab = obj.txt.index;
            obj.vocab{end + 1} = '<UNK>';
            obj.vocab{end + 1} = '<EOL>';
            obj.vocabSize = length(obj.vocab);
        end
        
        function [output,length] = fprop(obj,option)
            if nargin <= 1
                option = struct();
                option.reverse = false;
                option.fprop = true;
            end
            
            if ~isfield(option,'reverse')
                option.reverse = false;
            end
            
            if ~isfield(option,'fprop')
                option.fprop = true;
            end
            
            index = obj.base_point : obj.base_point + obj.batchSize  -  1;
            if option.fprop
                obj.base_point = mod(obj.base_point + obj.batchSize  -  1,size(obj.txt.data,1)) + 1;
            end
            prune_data = obj.txt.data(mod(index - 1,size(obj.txt.data,1)) + 1,:);
            mark = sum(prune_data,1) > 0;
            prune_data = prune_data(:,1 : sum(mark) + 1) + 1;
            for i = 1 : size(prune_data,1)
                prune_data(i,sum(prune_data(i,:) > 1) + 1) = obj.vocabSize;
            end
            if option.reverse
                prune_data = prune_data(:,end : -1 : 1);
            end
            obj.output = cell(2,size(prune_data,2));
            for i = 1 : size(prune_data,2)
                obj.output{1,i} = obj.convert.dataConvert(prune_data(:,i)');
                if obj.mask
                    obj.output{2,i} = obj.convert.dataConvert(prune_data(:,i)' > 1);
                else
                    obj.output{2,i} = obj.convert.dataConvert(ones(size(prune_data(:,i)')));
                end
            end
            output =  obj.output;
            length = size(output,2);
        end
        
        function initialOption(obj,option)
            if isfield(option,'batchSize')
                obj.batchSize = option.batchSize;
            end
            
            if isfield(option,'mask')
                obj.mask = option.mask;
            end
            
            if isfield(option,'base_point')
                obj.base_point = option.base_point;
            end
        end
    end
end