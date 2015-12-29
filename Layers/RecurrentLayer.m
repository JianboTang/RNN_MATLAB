classdef RecurrentLayer < OperateLayer
    properties
        W_T = Data();
        grad_W_T = Data();
        init_output
        grad_init_output
    end
    methods
        function obj = RecurrentLayer(option)
            if nargin == 0
                super_args{1} = struct();
            else if nargin == 1
                    super_args{1} = option;
                end
            end
            obj = obj@OperateLayer(super_args{:});
            obj.initialOption(super_args{:});
            obj.W_T = Data(super_args{:});
            obj.grad_W_T = Data(super_args{:});
            obj.initial();
        end
        
        function initial(obj)
            initial@OperateLayer(obj);
            obj.W_T.setDataSize([obj.hidden_dim,obj.hidden_dim]);
            obj.W_T.initial();
            if obj.backward
                obj.grad_W_T.setDataSize([obj.hidden_dim,obj.hidden_dim]);
                obj.grad_W_T.setZeros();
            end
        end
       
        function [output,length] = fprop(obj,input,length)
            if obj.debug
                if isempty(obj.W.context) || isempty(obj.W_T.context) || isempty(obj.B.context)
                    error('not all the context are initialized yet,and can not do the fprop operation!');
                end
            end
            obj.length = length;
            obj.batch_size = size(input{1,1},2);
            if isempty(obj.init_output)
                obj.init.setDataSize([obj.hidden_dim,obj.batch_size]);
                obj.init.setZeros();
                obj.init_output{1,1} = obj.init.context;
                obj.init.clearData();
            end
            for i  = 1 : size(input,2)
                obj.input{1,i} = input{1,i};
                if i == 1
                    obj.output{1,i} = obj.activation(bsxfun(@plus,obj.W.context * obj.input{1,i} + obj.W_T.context * obj.init_output{1,1},obj.B.context));
                else
                    obj.output{1,i} = obj.activation(bsxfun(@plus,obj.W.context * obj.input{1,i} + obj.W_T.context * obj.output{1,i - 1},obj.B.context));
                end
            end
            output = obj.output;
        end
       
        function output = fprop_step(obj,input,i)
            if obj.debug
                if isempty(obj.W.context) || isempty(obj.W_T.context) || isempty(obj.B.context)
                    error('not all the context are initialized yet,and can not do the fprop operation!');
                end
            end
            obj.length = i;
            obj.batch_size = size(input{1,1},2);
            if isempty(obj.init_output)
                obj.init.setDataSize([obj.hidden_dim,obj.batch_size]);
                obj.init.setZeros();
                obj.init_output{1,1} = obj.init.context;
                obj.init.clearData();
            end
            if i == 1
                obj.output{1,i} = obj.activation(bsxfun(@plus,obj.W.context * input{1,1} + obj.W_T.context * obj.init_output{1,1},obj.B.context));
            else
                obj.output{1,i} = obj.activation(bsxfun(@plus,obj.W.context * input{1,1} + obj.W_T.context * obj.output{1,i - 1},obj.B.context));
            end
            output{1,1} = obj.output{1,i};
        end
        
        function initialOption(obj,option)
            initialOption@OperateLayer(obj,option);
        end
        
        function grad_input = bprop(obj,grad_output)
            for i = obj.length : -1 : 1
                obj.grad_output{1,i} = grad_output{1,i};
                if i == size(obj.grad_output,2)
                    obj.grad_input{1,i} = obj.grad_output{1,i} .* obj.diff_activ(obj.output{1,i});
                else
                    obj.grad_input{1,i} = (obj.grad_output{1,i} + obj.W_T.context' * obj.grad_input{1,i + 1}) .* obj.diff_activ(obj.output{1,i});
                end
                if i > 1
                    obj.grad_W_T.context = obj.grad_W_T.context + obj.grad_input{1,i} * (obj.output{1,i - 1})' ./ obj.batch_size;
                else
                    obj.grad_W_T.context = obj.grad_W_T.context + obj.grad_input{1,i} * (obj.init_output{1,1})' ./ obj.batch_size;
                end
            end
            obj.grad_init_output{1,1} = obj.W_T.context' * obj.grad_input{1,1};
            for i = 1 : size(obj.grad_input,2)
                obj.grad_B.context = obj.grad_B.context + sum(obj.grad_input{1,i},2) ./ obj.batch_size;
                obj.grad_W.context = obj.grad_W.context + obj.grad_input{1,i} * (obj.input{1,i})' ./ obj.batch_size;
                obj.grad_input{1,i} = obj.W.context' * obj.grad_input{1,i};
            end
            grad_input = obj.grad_input;
        end
        
        function update(obj,apply,option)
            if nargin <= 2
                option = struct();
            end
            obj.W.context = apply(obj.W.context,obj.grad_W.context,option);
            obj.W_T.context = apply(obj.W_T.context,obj.grad_W_T.context,option);
            obj.B.context = apply(obj.B.context,obj.grad_B.context,option);
            
            obj.grad_W.setZeros();
            obj.grad_W_T.setZeros();
            obj.grad_B.setZeros();
        end
        
        function object = saveObj(obj)
            if obj.W.useGPU
                object.W = gather(obj.W.context);
                object.W_T = gather(obj.W_T.context);
                object.B = gather(obj.B.context);
            else
                object.W = obj.W.context;
                object.W_T = obj.W_T.context;
                object.B = obj.B.context;
            end
            
            object.activation = obj.activation;
            object.diff_activ = obj.diff_activ;
        end
        
        function loadObj(obj,object)
            obj.W.context = obj.init.dataConvert(object.W);
            obj.B.context = obj.init.dataConvert(object.B);
            obj.W_T.context = obj.init.dataConvert(object.W_T);
            obj.activation = object.activaiton;
            obj.diff_activ = object.diff_activ;
        end
        %% the functions below this line are used in the above ones or some are just defined for the gradient check;
        function checkGrad(obj)
            seqLen = 10;
            batchSize = 20;
            input = cell([1,seqLen]);
            target = cell([1,seqLen]);
            for i = 1 : seqLen
                input{1,i} = randn([obj.input_dim,batchSize]);
                target{1,i} = randn([obj.hidden_dim,batchSize]);
            end
            epislon = 10 ^ (-6);
            
            W = obj.W.context;
            W_T = obj.W_T.context;
            B = obj.B.context;
            
            obj.fprop(input,size(input,2));
            grad_output = cell([1,size(obj.output,2)]);
            for i = 1 : size(obj.output,2)
                grad_output{1,i} = 2 * (obj.output{1,i} - target{1,i});
            end
            obj.bprop(grad_output);
            
            grad_input = obj.grad_input;
            grad_W = obj.grad_W.context;
            grad_W_T = obj.grad_W_T.context;
            grad_B = obj.grad_B.context;
            init_output = obj.init_output{1,1};
            grad_init_output = obj.grad_init_output{1,1};
            numeric_grad_W = zeros(size(W));
            numeric_grad_W_T = zeros(size(W_T));
            numeric_grad_B = zeros(size(grad_B));
            numeric_grad_input = cell(size(grad_input));
            numeric_grad_init_output = zeros(size(grad_init_output));
            for i = 1 : size(numeric_grad_input,2)
                numeric_grad_input{1,i} = zeros(size(grad_input{1,i}));
            end
            %% the W parameter check
            for n = 1 : size(W,1)
                for m = 1 : size(W,2)
                    obj.W.context = W;
                    obj.W.context(n,m) = obj.W.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W.context = W;
                    obj.W.context(n,m) = obj.W.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W(:) - grad_W(:)) ./ norm(numeric_grad_W(:) + grad_W(:));
            if obj.debug
                disp([numeric_grad_W(:),obj.grad_W.context(:)]);
            end
            disp(['the W parameter check is ' , num2str(norm_diff)])
            
            %%  the W_T parameter check
            for n = 1 : size(W_T,1)
                for m = 1 : size(W_T,2)
                    obj.W_T.context = W_T;
                    obj.W_T.context(n,m) = obj.W_T.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_T.context = W_T;
                    obj.W_T.context(n,m) = obj.W_T.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_T(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_T(:) - grad_W_T(:)) ./ norm(numeric_grad_W_T(:) + grad_W_T(:));
            if obj.debug
                disp([numeric_grad_W_T(:),grad_W_T(:)]);
            end
            disp(['the W_T parameter check is ' , num2str(norm_diff)])
            
            %% the B parameter check
            for n = 1 : size(B,1)
                for m = 1 : size(B,2)
                    obj.B.context = B;
                    obj.B.context(n,m) = obj.B.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.B.context = B;
                    obj.B.context(n,m) = obj.B.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_B(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_B(:) - grad_B(:)) ./ norm(numeric_grad_B(:) + grad_B(:));
            if obj.debug
                disp([numeric_grad_B(:),grad_B(:)]);
            end
            disp(['the B parameter check is ' , num2str(norm_diff)])
            
            %%  the init_output parameter check
            for n = 1 : size(init_output,1)
                for m = 1 : size(init_output,2)
                    temp = init_output;
                    temp(n,m) = temp(n,m) + epislon;
                    obj.init_output{1,1} = temp;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    temp = init_output;
                    temp(n,m) = temp(n,m) - epislon;
                    obj.init_output{1,1} = temp;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_init_output(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_init_output(:) - grad_init_output(:) ./ batchSize) ./ norm(numeric_grad_init_output(:) + grad_init_output(:) ./ batchSize);
            if obj.debug
                disp([numeric_grad_init_output(:),grad_init_output(:)]);
            end
            disp(['the init_output parameter check is ' , num2str(norm_diff)])
            %% check the gradient of input data
            for t = 1 : seqLen
                temp = input{1,t};
                for i = 1 : size(temp,1)
                    for j = 1 : size(temp,2)
                        temp_input = input;
                        temp = temp_input{1,t};
                        temp(i,j) = temp(i,j) + epislon;
                        temp_input{1,t} = temp;
                        obj.fprop(temp_input,size(temp_input,2));
                        cost_1 = getCost(target,obj.output);

                        temp_input = input;
                        temp = temp_input{1,t};
                        temp(i,j) = temp(i,j) - epislon;
                        temp_input{1,t} = temp;
                        obj.fprop(temp_input,size(temp_input,2));
                        cost_2 = getCost(target,obj.output);
                        numeric_grad_input{1,t}(i,j) = (cost_1 - cost_2) ./ (2 * epislon);
                    end
                end
                norm_diff = norm(numeric_grad_input{1,t}(:) - grad_input{1,t}(:) ./ batchSize) ./ norm(numeric_grad_input{1,t}(:) + grad_input{1,t}(:) ./ batchSize);
                if obj.debug
                    disp([numeric_grad_input{1,t}(:),grad_input{1,t}(:)]);
                end
                disp([num2str(t),' : the check of input gradient is ' , num2str(norm_diff)])
            end
        end
    end
end

function cost = getCost(target,output)
    cost = 0;
    for m = 1 : size(target,2)
        temp = (target{1,m} - output{1,m}) .^ 2;
        cost = cost + sum(temp(:)) ./ size(temp,2);
    end
end