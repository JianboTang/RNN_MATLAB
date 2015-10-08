classdef SoftmaxLayer < OperateLayer
    methods
        function obj = SoftmaxLayer(option)
            if nargin == 0
                super_args{1} = struct();
            else if nargin == 1
                    super_args{1} = option;
                end
            end
            obj = obj@OperateLayer(super_args{:});
        end
        
        function forward(obj)
            if obj.debug
                if isempty(obj.inputData.data)
                    error('you should set the inputData first');
                end
            end
            obj.outputData.setData(obj.W.data * obj.inputData.data);
            obj.outputData.setData(exp(bsxfun(@minus,obj.outputData.data,max(obj.outputData.data,[],1))));
            obj.outputData.setData(bsxfun(@rdivide,obj.outputData.data,sum(obj.outputData.data,1)));
        end
    end
end