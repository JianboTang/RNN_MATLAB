classdef LossLayer < handle
    properties
        target = DataLayer();
        cost = [];
    end
    methods
        function obj = LossLayer(option)
            if nargin >= 1
                obj.target = DataLayer(option);
            end
        end
        
        function setTarget(obj,target)
            obj.target.data = target;
        end
    end
        %% the functions below this line are used in the above functions 
end