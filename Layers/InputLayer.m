classdef InputLayer < handle
    properties
        txt = textData('../Data/Text/test_small.txt','../Data/Text/dictionary');
        output
        base_point = 1
        batchSize
        convert = Data();
        vocabSize
        vocab
        used_index
    end
    methods
        function obj = InputLayer(option)
            if nargin >= 1
                obj.initialOption(option);
                obj.convert = Data(option);
            end
            obj.vocab = obj.txt.index;
            obj.vocabSize = length(obj.vocab);
        end
        
        function getFile(obj,txtfile,dictionary)
            if ~exist(txtfile,'file')
                error([txtfile,'do not found !']);
            end
            
            if ~exist(dictionary,'file')
                error([dictionary,'do not found']);
            end
            obj.txt = textData(txtfile,dictionary);
            obj.vocab = obj.txt.index;
            obj.vocabSize = length(obj.vocab);
        end
        
        function [output,length] = fprop(obj,option)
            if nargin <= 1
                option = struct();
                option.reverse = false;
                option.fprop = true;
                option.rightAlig = false;
            end
            
            if ~isfield(option,'reverse')
                option.reverse = false;
            end
            
            if ~isfield(option,'fprop')
                option.fprop = true;
            end
            
            if ~isfield(option,'rightAlig')
                option.rightAlig = false;
            end
            
            if isempty(obj.batchSize)
                if isempty(obj.txt.data)
                    error('you should appoint the txtfile of txt first');
                end
                obj.batchSize = size(obj.txt.data,1);
            end
            
            index = obj.base_point : obj.base_point + obj.batchSize  -  1;
            obj.used_index = mod(index - 1,size(obj.txt.data,1)) + 1;
            prune_data = obj.txt.data(obj.used_index,:);
            mark = sum(prune_data,1) > 0;
            prune_data = prune_data(:,1 : sum(mark));
            
            if option.reverse
                prune_data = prune_data(:,end : -1 : 1);
            end
            
            if option.rightAlig
                new_data = zeros(size(prune_data));
                for i = 1 : size(prune_data,1)
                    text = prune_data(i,:);
                    zeros_index = find(text == 0);
                    if isempty(zeros_index)
                        new_data(i,:) = text;
                        continue;
                    end
                    offset = size(text,2) - zeros_index(1,1) + 2;
                    new_data(i,offset : end) = text(1,1 : zeros_index(1,1) - 1);
                end
                prune_data = new_data;
            end
            
            prune_data(prune_data == 0) = 1;
            
            obj.output = cell(1,size(prune_data,2));
            for i = 1 : size(prune_data,2)
                obj.output{1,i} = obj.convert.dataConvert(prune_data(:,i)');
            end
            
            if option.fprop
                obj.base_point = mod(obj.base_point + obj.batchSize  -  1,size(obj.txt.data,1)) + 1;
            end
            
            output =  obj.output;
            length = size(output,2);
        end
        
        function initialOption(obj,option)
            if isfield(option,'batchSize')
                obj.batchSize = option.batchSize;
            end
            
            if isfield(option,'base_point')
                obj.base_point = option.base_point;
            end
        end
    end
end