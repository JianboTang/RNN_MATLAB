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
        end
        
        function forward(obj)
            obj.outputData.setData(reshape(obj.W.data(:,obj.inputData.data(:)),[size(obj.W.data,1),size(obj.inputData.data)]));
        end
    end
end