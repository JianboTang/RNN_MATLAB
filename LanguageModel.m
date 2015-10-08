classdef LanguageModel < handle
    properties
        input = InputLayer();
        wordEmbedd = EmbeddingLayer();
        hidden = RecurrentLayer();
        output = SoftmaxLayer();
    end
    properties (SetAccess = private,GetAccess = public,Hidden)
        txtFile = ''
        dictionary = ''
        embeddingDimension = 16;
    end
    methods 
        function initialOption(obj,option)
            if isfield(option,'hiddenNetSize')
                obj.setHiddenNetSize(option.hiddenNetSize);
            end
            
            if isfield(option,'txtFile')
                obj.setTxtFile(option.txtFile)
            end
            
            if isfield(option,'dictionary')
                obj.setDictionary(option.dictioanry);
            end
            
            if isfield(option,'embeddingDimension')
                obj.setEmbeddingDimension(option.embeddingDimension);
            end
        end
        
        function setEmbeddingDimension(obj,embeddingDimension)
            obj.embeddingDimension = embeddingDimension;
        end
        
        function setHiddenNetSize(obj,hiddenNetSize)
            obj.hiddenNetSize = hiddenNetSize;
        end
        
        function setDictionary(obj,dictionary)
            obj.dictionary = dictionary;
        end
        
        function setTxtFile(obj,txtFile)
            obj.txtFile = txtFile;
        end
        
        
        function initial(obj)
            obj.input.forward();
            obj.wordEmbedd.setDimension([obj.embeddingDimension,size(obj.input.data.index,1) + 1]);
            obj.wordEmbedd.initial();
            
            for i = 1 : size(obj.hidden,2)
                if i == 1
                    obj.hidden(1,i).setDimension([obj.hiddenNetSize(1,i),obj.embeddingDimension]);
                    obj.hidden(1,i).initial();
                else
                    obj.hidden(1,i).setDimension([obj.hiddenNetSize(1,i),obj.hiddenNetSize(1,i - 1)]);
                    obj.hidden(1,i).initial();
                end
            end
            
            obj.output.setDimension([obj.hiddenNetSize(1,i),size(obj.input.data.index,1) + 1]);
            obj.output.initial();
        end
    end
end