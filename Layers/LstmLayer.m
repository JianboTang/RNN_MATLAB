classdef LstmLayer < OperateLayer
    %this layer is an implemention of lstm unit mentioned in paper <supervised sequence labelling
    %with recurrent neural networks>,page 38
    properties
        W_il = Data();
        W_hl = Data();
        W_cl = Data();
        W_cf = Data();
        W_hf = Data();
        W_if = Data();
        W_hc = Data();
        W_ic = Data();
        W_cw = Data();
        W_hw = Data();
        W_iw = Data();
        B_l  = Data();
        B_f  = Data();
        B_c  = Data();
        B_w  = Data();
        grad_W_il = Data();
        grad_W_hl = Data();
        grad_W_cl = Data();
        grad_W_cf = Data();
        grad_W_hf = Data();
        grad_W_if = Data();
        grad_W_hc = Data();
        grad_W_ic = Data();
        grad_W_cw = Data();
        grad_W_hw = Data();
        grad_W_iw = Data();
        grad_B_l  = Data();
        grad_B_f  = Data();
        grad_B_c  = Data();
        grad_B_w  = Data();
        gate_activation = @sigmoid
        gate_diff_activ = @sigmoidPrime
        cell_activation = @tanh
        cell_diff_activ = @tanhPrime
        cells
        states
        input_gates
        forget_gates
        output_gates
        grad_cells
        grad_states
        grad_input_gates
        grad_forget_gates
        grad_output_gates
        init_output
        init_state
        grad_init_output
        grad_init_state
    end
    methods 
        function obj = LstmLayer(option)
            if nargin == 0
                super_args{1} = struct();
            else if nargin == 1
                    super_args{1} = option;
                end
            end
            obj = obj@OperateLayer(super_args{:});
            obj.initialOption(super_args{:});
            obj.W_il = Data(super_args{:});
            obj.W_hl = Data(super_args{:});
            obj.W_cl = Data(super_args{:});
            obj.W_cf = Data(super_args{:});
            obj.W_hf = Data(super_args{:});
            obj.W_if = Data(super_args{:});
            obj.W_hc = Data(super_args{:});
            obj.W_ic = Data(super_args{:});
            obj.W_cw = Data(super_args{:});
            obj.W_hw = Data(super_args{:});
            obj.W_iw = Data(super_args{:});
            obj.B_l  = Data(super_args{:});
            obj.B_f  = Data(super_args{:});
            obj.B_c  = Data(super_args{:});
            obj.B_w  = Data(super_args{:});
            obj.grad_W_il = Data(super_args{:});
            obj.grad_W_hl = Data(super_args{:});
            obj.grad_W_cl = Data(super_args{:});
            obj.grad_W_cf = Data(super_args{:});
            obj.grad_W_hf = Data(super_args{:});
            obj.grad_W_if = Data(super_args{:});
            obj.grad_W_hc = Data(super_args{:});
            obj.grad_W_ic = Data(super_args{:});
            obj.grad_W_cw = Data(super_args{:});
            obj.grad_W_hw = Data(super_args{:});
            obj.grad_W_iw = Data(super_args{:});
            obj.grad_B_l  = Data(super_args{:});
            obj.grad_B_f  = Data(super_args{:});
            obj.grad_B_c  = Data(super_args{:});
            obj.grad_B_w  = Data(super_args{:});
            obj.initial();
        end
        
        function initial(obj)
            obj.W_il.setDataSize([obj.hidden_dim,obj.input_dim]);
            obj.W_il.initial();
            obj.W_hl.setDataSize([obj.hidden_dim,obj.hidden_dim]);
            obj.W_hl.initial();
            obj.W_cl.setDataSize([obj.hidden_dim,obj.hidden_dim]);
            obj.W_cl.initial();
            obj.W_cf.setDataSize([obj.hidden_dim,obj.hidden_dim]);
            obj.W_cf.initial();
            obj.W_hf.setDataSize([obj.hidden_dim,obj.hidden_dim]);
            obj.W_hf.initial();
            obj.W_if.setDataSize([obj.hidden_dim,obj.input_dim]);
            obj.W_if.initial();
            obj.W_hc.setDataSize([obj.hidden_dim,obj.hidden_dim]);
            obj.W_hc.initial();
            obj.W_ic.setDataSize([obj.hidden_dim,obj.input_dim]);
            obj.W_ic.initial();
            obj.W_cw.setDataSize([obj.hidden_dim,obj.hidden_dim]);
            obj.W_cw.initial();
            obj.W_hw.setDataSize([obj.hidden_dim,obj.hidden_dim]);
            obj.W_hw.initial();
            obj.W_iw.setDataSize([obj.hidden_dim,obj.input_dim]);
            obj.W_iw.initial();
            obj.B_l.setDataSize([obj.hidden_dim,1]);
            obj.B_l.initial();
            obj.B_f.setDataSize([obj.hidden_dim,1]);
            obj.B_f.initial();
            obj.B_c.setDataSize([obj.hidden_dim,1]);
            obj.B_c.initial();
            obj.B_w.setDataSize([obj.hidden_dim,1]);
            obj.B_w.initial();
            if obj.backward
                obj.grad_W_il.setDataSize([obj.hidden_dim,obj.input_dim]);
                obj.grad_W_il.setZeros();
                obj.grad_W_hl.setDataSize([obj.hidden_dim,obj.hidden_dim]);
                obj.grad_W_hl.setZeros();
                obj.grad_W_cl.setDataSize([obj.hidden_dim,obj.hidden_dim]);
                obj.grad_W_cl.setZeros();
                obj.grad_W_cf.setDataSize([obj.hidden_dim,obj.hidden_dim]);
                obj.grad_W_cf.setZeros();
                obj.grad_W_hf.setDataSize([obj.hidden_dim,obj.hidden_dim]);
                obj.grad_W_hf.setZeros();
                obj.grad_W_if.setDataSize([obj.hidden_dim,obj.input_dim]);
                obj.grad_W_if.setZeros();
                obj.grad_W_hc.setDataSize([obj.hidden_dim,obj.hidden_dim]);
                obj.grad_W_hc.setZeros();
                obj.grad_W_ic.setDataSize([obj.hidden_dim,obj.input_dim]);
                obj.grad_W_ic.setZeros();
                obj.grad_W_cw.setDataSize([obj.hidden_dim,obj.hidden_dim]);
                obj.grad_W_cw.setZeros();
                obj.grad_W_hw.setDataSize([obj.hidden_dim,obj.hidden_dim]);
                obj.grad_W_hw.setZeros();
                obj.grad_W_iw.setDataSize([obj.hidden_dim,obj.input_dim]);
                obj.grad_W_iw.setZeros();
                obj.grad_B_l.setDataSize([obj.hidden_dim,1]);
                obj.grad_B_l.setZeros();
                obj.grad_B_f.setDataSize([obj.hidden_dim,1]);
                obj.grad_B_f.setZeros();
                obj.grad_B_c.setDataSize([obj.hidden_dim,1]);
                obj.grad_B_c.setZeros();
                obj.grad_B_w.setDataSize([obj.hidden_dim,1]);
                obj.grad_B_w.setZeros();
            end
        end
        
        function [output,length] = fprop(obj,input,length)
            if obj.debug
                if isempty(obj.W_il.context) || isempty(obj.W_hl.context) || isempty(obj.W_cl.context) || ...
                        isempty(obj.W_cf.context) || isempty(obj.W_hf.context) || isempty(obj.W_if.context) || ...
                        isempty(obj.W_hc.context) || isempty(obj.W_ic.context) || isempty(obj.W_cw.context) || ...
                        isempty(obj.W_hw.context) || isempty(obj.W_iw.context) || isempty(obj.B_l.context) || ...
                        isempty(obj.B_f.context) || isempty(obj.B_c.context) || isempty(obj.B_w.context)
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
                obj.init.setDataSize([1,obj.batch_size]);
                obj.init.setOnes();
                obj.init_output{2,1} = obj.init.context;
                obj.init.clearData();
            end
            if isempty(obj.init_state)
                obj.init.setDataSize([obj.hidden_dim,obj.batch_size]);
                obj.init.setZeros();
                obj.init_state{1,1} = obj.init.context;
                obj.init.clearData();
            end
            
            for i = 1 : obj.length
                obj.input{1,i} = input{1,i};
                obj.input{2,i} = input{2,i};
                if i == 1
                    obj.input_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_il.context * obj.input{1,i} + obj.W_hl.context * obj.init_output{1,1} ...
                        + obj.W_cl.context * obj.init_state{1,1},obj.B_l.context));
                    obj.input_gates{1,i} = bsxfun(@times,obj.input_gates{1,i},obj.input{2,i});
                    obj.forget_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_if.context * obj.input{1,i} + obj.W_hf.context * obj.init_output{1,1} ...
                        + obj.W_cf.context * obj.init_state{1,1},obj.B_f.context));
                    obj.forget_gates{1,i} = bsxfun(@times,obj.forget_gates{1,i},obj.input{2,i});
                    obj.cells{1,i} =obj.cell_activation(bsxfun(@plus,obj.W_ic.context * obj.input{1,i} + obj.W_hc.context * ...
                        obj.init_output{1,1},obj.B_c.context));
                    obj.cells{1,i} = bsxfun(@times,obj.cells{1,i},obj.input{2,i});
                    obj.states{1,i} = obj.forget_gates{1,i} .* obj.init_state{1,1} +  obj.input_gates{1,i} .* obj.cells{1,i};
                    obj.states{1,i} = bsxfun(@times,obj.states{1,i},obj.input{2,i});
                    obj.output_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_iw.context * obj.input{1,i} + obj.W_hw.context * obj.init_output{1,1} ...
                        + obj.W_cw.context * obj.states{1,i},obj.B_w.context));
                    obj.output_gates{1,i} = bsxfun(@times,obj.output_gates{1,i},obj.input{2,i});
                else
                    obj.input_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_il.context * obj.input{1,i} + ...
                        obj.W_hl.context * obj.output{1,i - 1} + obj.W_cl.context * obj.states{1,i - 1},obj.B_l.context));
                    obj.input_gates{1,i} = bsxfun(@times,obj.input_gates{1,i},obj.input{2,i});
                    obj.forget_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_if.context * obj.input{1,i} + ...
                        obj.W_hf.context * obj.output{1,i - 1} + obj.W_cf.context * obj.states{1,i - 1},obj.B_f.context));
                    obj.forget_gates{1,i} = bsxfun(@times,obj.forget_gates{1,i},obj.input{2,i});
                    obj.cells{1,i} = obj.cell_activation(bsxfun(@plus,obj.W_ic.context * obj.input{1,i} + obj.W_hc.context * obj.output{1,i - 1},obj.B_c.context));
                    obj.cells{1,i} = bsxfun(@times,obj.cells{1,i},obj.input{2,i});
                    obj.states{1,i} =  obj.forget_gates{1,i} .* obj.states{1,i - 1} + obj.input_gates{1,i} .* obj.cells{1,i};
                    obj.states{1,i} = bsxfun(@times,obj.states{1,i},obj.input{2,i});
                    obj.output_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_iw.context * obj.input{1,i} + ...
                        obj.W_hw.context * obj.output{1,i - 1} + obj.W_cw.context * obj.states{1,i},obj.B_w.context));
                    obj.output_gates{1,i} = bsxfun(@times,obj.output_gates{1,i},obj.input{2,i});
                end
                obj.output{1,i} = obj.output_gates{1,i} .* obj.activation(obj.states{1,i});
                obj.output{1,i} = bsxfun(@times,obj.output{1,i},obj.input{2,i});
                obj.output{2,i} = obj.input{2,i};
            end
            output = obj.output;
            if obj.debug
                display(['LstmLayer | W_il | mean : ',num2str(mean(obj.W_il.context(:))),' | std : ',num2str(std(obj.W_il.context(:)))]);
                display(['LstmLayer | W_hl | mean : ',num2str(mean(obj.W_hl.context(:))),' | std : ',num2str(std(obj.W_hl.context(:)))]);
                display(['LstmLayer | W_cl | mean : ',num2str(mean(obj.W_cl.context(:))),' | std : ',num2str(std(obj.W_cl.context(:)))]);
                display(['LstmLayer | W_cf | mean : ',num2str(mean(obj.W_cf.context(:))),' | std : ',num2str(std(obj.W_cf.context(:)))]);
                display(['LstmLayer | W_hf | mean : ',num2str(mean(obj.W_hf.context(:))),' | std : ',num2str(std(obj.W_hf.context(:)))]);
                display(['LstmLayer | W_if | mean : ',num2str(mean(obj.W_if.context(:))),' | std : ',num2str(std(obj.W_if.context(:)))]);
                display(['LstmLayer | W_hc | mean : ',num2str(mean(obj.W_hc.context(:))),' | std : ',num2str(std(obj.W_hc.context(:)))]);
                display(['LstmLayer | W_ic | mean : ',num2str(mean(obj.W_ic.context(:))),' | std : ',num2str(std(obj.W_ic.context(:)))]);
                display(['LstmLayer | W_cw | mean : ',num2str(mean(obj.W_cw.context(:))),' | std : ',num2str(std(obj.W_cw.context(:)))]);
                display(['LstmLayer | W_hw | mean : ',num2str(mean(obj.W_hw.context(:))),' | std : ',num2str(std(obj.W_hw.context(:)))]);
                display(['LstmLayer | W_iw | mean : ',num2str(mean(obj.W_iw.context(:))),' | std : ',num2str(std(obj.W_iw.context(:)))]);
            end
        end
        
        function output = fprop_step(obj,input,i)
            if obj.debug
                if isempty(obj.W_il.context) || isempty(obj.W_hl.context) || isempty(obj.W_cl.context) || ...
                        isempty(obj.W_cf.context) || isempty(obj.W_hf.context) || isempty(obj.W_if.context) || ...
                        isempty(obj.W_hc.context) || isempty(obj.W_ic.context) || isempty(obj.W_cw.context) || ...
                        isempty(obj.W_hw.context) || isempty(obj.W_iw.context) || isempty(obj.B_l.context) || ...
                        isempty(obj.B_f.context) || isempty(obj.B_c.context) || isempty(obj.B_w.context)
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
                obj.init.setDataSize([1,obj.batch_size]);
                obj.init.setOnes();
                obj.init_output{2,1} = obj.init.context;
                obj.init.clearData();
            end
            if isempty(obj.init_state)
                obj.init.setDataSize([obj.hidden_dim,obj.batch_size]);
                obj.init.setZeros();
                obj.init_state{1,1} = obj.init.context;
                obj.init.clearData();
            end
            
            obj.input{1,i} = input{1,1};
            obj.input{2,i} = input{2,1};
            if i == 1
                obj.input_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_il.context * obj.input{1,i} + obj.W_hl.context * obj.init_output{1,1} ...
                    + obj.W_cl.context * obj.init_state{1,1},obj.B_l.context));
                obj.input_gates{1,i} = bsxfun(@times,obj.input_gates{1,i},obj.input{2,i});
                obj.forget_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_if.context * obj.input{1,i} + obj.W_hf.context * obj.init_output{1,1} ...
                    + obj.W_cf.context * obj.init_state{1,1},obj.B_f.context));
                obj.forget_gates{1,i} = bsxfun(@times,obj.forget_gates{1,i},obj.input{2,i});
                obj.cells{1,i} =obj.cell_activation(bsxfun(@plus,obj.W_ic.context * obj.input{1,i} + obj.W_hc.context * ...
                    obj.init_output{1,1},obj.B_c.context));
                obj.cells{1,i} = bsxfun(@times,obj.cells{1,i},obj.input{2,i});
                obj.states{1,i} = obj.forget_gates{1,i} .* obj.init_state{1,1} +  obj.input_gates{1,i} .* obj.cells{1,i};
                obj.states{1,i} = bsxfun(@times,obj.states{1,i},obj.input{2,i});
                obj.output_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_iw.context * obj.input{1,i} + obj.W_hw.context * obj.init_output{1,1} ...
                    + obj.W_cw.context * obj.states{1,i},obj.B_w.context));
                obj.output_gates{1,i} = bsxfun(@times,obj.output_gates{1,i},obj.input{2,i});
            else
                obj.input_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_il.context * obj.input{1,i} + ...
                    obj.W_hl.context * obj.output{1,i - 1} + obj.W_cl.context * obj.states{1,i - 1},obj.B_l.context));
                obj.input_gates{1,i} = bsxfun(@times,obj.input_gates{1,i},obj.input{2,i});
                obj.forget_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_if.context * obj.input{1,i} + ...
                    obj.W_hf.context * obj.output{1,i - 1} + obj.W_cf.context * obj.states{1,i - 1},obj.B_f.context));
                obj.forget_gates{1,i} = bsxfun(@times,obj.forget_gates{1,i},obj.input{2,i});
                obj.cells{1,i} = obj.cell_activation(bsxfun(@plus,obj.W_ic.context * obj.input{1,i} + obj.W_hc.context * obj.output{1,i - 1},obj.B_c.context));
                obj.cells{1,i} = bsxfun(@times,obj.cells{1,i},obj.input{2,i});
                obj.states{1,i} =  obj.forget_gates{1,i} .* obj.states{1,i - 1} + obj.input_gates{1,i} .* obj.cells{1,i};
                obj.states{1,i} = bsxfun(@times,obj.states{1,i},obj.input{2,i});
                obj.output_gates{1,i} = obj.gate_activation(bsxfun(@plus,obj.W_iw.context * obj.input{1,i} + ...
                    obj.W_hw.context * obj.output{1,i - 1} + obj.W_cw.context * obj.states{1,i},obj.B_w.context));
                obj.output_gates{1,i} = bsxfun(@times,obj.output_gates{1,i},obj.input{2,i});
            end
            obj.output{1,i} = obj.output_gates{1,i} .* obj.activation(obj.states{1,i});
            obj.output{1,i} = bsxfun(@times,obj.output{1,i},obj.input{2,i});
            obj.output{2,i} = obj.input{2,i};
            
            output{1,1} = obj.output{1,i};
            output{2,1} = obj.output{2,i};
        end
        
        function grad_input = bprop(obj,grad_output)
            for i = obj.length : -1 : 1
                obj.grad_output{1,i} = grad_output{1,i};
                if i == obj.length
                    obj.grad_output{1,i} = bsxfun(@times,obj.grad_output{1,i},obj.output{2,i});
                    obj.grad_output_gates{1,i} = obj.gate_diff_activ(obj.output_gates{1,i}) .* obj.activation(obj.states{1,i}) .* obj.grad_output{1,i};
                    obj.grad_output_gates{1,i} = bsxfun(@times,obj.grad_output_gates{1,i},obj.output{2,i});
                    obj.grad_states{1,i} = obj.output_gates{1,i} .* obj.diff_activ(obj.activation(obj.states{1,i})) .* ...
                        obj.grad_output{1,i} + obj.W_cw.context' * obj.grad_output_gates{1,i};% obj.grad_states{1,i} + 
                    obj.grad_states{1,i} = bsxfun(@times,obj.grad_states{1,i},obj.output{2,i});
                    obj.grad_cells{1,i} = obj.input_gates{1,i} .* obj.cell_diff_activ(obj.cells{1,i}) .* obj.grad_states{1,i};
                    obj.grad_cells{1,i} = bsxfun(@times,obj.grad_cells{1,i},obj.output{2,i});
                    obj.grad_forget_gates{1,i} = obj.gate_diff_activ(obj.forget_gates{1,i}) .* obj.states{1,i - 1} .* obj.grad_states{1,i};  
                    obj.grad_forget_gates{1,i} = bsxfun(@times,obj.grad_forget_gates{1,i},obj.output{2,i});
                    obj.grad_input_gates{1,i} = obj.gate_diff_activ(obj.input_gates{1,i}) .* obj.cells{1,i} .* obj.grad_states{1,i};
                    obj.grad_input_gates{1,i} = bsxfun(@times,obj.grad_input_gates{1,i},obj.output{2,i});
                    obj.grad_input{1,i} = obj.W_il.context' * obj.grad_input_gates{1,i} + obj.W_if.context' * obj.grad_forget_gates{1,i} + ...
                        obj.W_ic.context' * obj.grad_cells{1,i} + obj.W_iw.context' * obj.grad_output_gates{1,i};
                else
                    obj.grad_output{1,i} = obj.grad_output{1,i} + obj.W_hl.context' * obj.grad_input_gates{1,i + 1} + ...
                        obj.W_hf.context' * obj.grad_forget_gates{1,i + 1} ...
                        + obj.W_hc.context' * obj.grad_cells{1,i + 1} + obj.W_hw.context' * obj.grad_output_gates{1,i + 1};
                    obj.grad_output{1,i} = bsxfun(@times,obj.grad_output{1,i},obj.output{2,i});
                    obj.grad_output_gates{1,i} = obj.gate_diff_activ(obj.output_gates{1,i}) .* obj.activation(obj.states{1,i}) .* obj.grad_output{1,i};
                    obj.grad_output_gates{1,i} = bsxfun(@times,obj.grad_output_gates{1,i},obj.output{2,i});
                    obj.grad_states{1,i} = obj.output_gates{1,i} .* obj.diff_activ(obj.activation(obj.states{1,i})) .* obj.grad_output{1,i} + ...
                        obj.W_cw.context' * obj.grad_output_gates{1,i} + obj.forget_gates{1,i + 1} .* obj.grad_states{1,i + 1} + ...
                        obj.W_cf.context' * obj.grad_forget_gates{1,i + 1} ...
                        + obj.W_cl.context' * obj.grad_input_gates{1,i + 1};%obj.grad_states{1,i} + 
                    obj.grad_states{1,i} = bsxfun(@times,obj.grad_states{1,i},obj.output{2,i});
                    obj.grad_cells{1,i} = obj.input_gates{1,i} .* obj.cell_diff_activ(obj.cells{1,i}) .* obj.grad_states{1,i};
                    obj.grad_cells{1,i} = bsxfun(@times,obj.grad_cells{1,i},obj.output{2,i});
                    obj.grad_input_gates{1,i} = obj.gate_diff_activ(obj.input_gates{1,i}) .* obj.cells{1,i} .* obj.grad_states{1,i};
                    obj.grad_input_gates{1,i} = bsxfun(@times,obj.grad_input_gates{1,i},obj.output{2,i});
                    if i > 1
                        obj.grad_forget_gates{1,i} = obj.gate_diff_activ(obj.forget_gates{1,i}) .* obj.states{1,i - 1} .* obj.grad_states{1,i}; 
                        obj.grad_forget_gates{1,i} = bsxfun(@times,obj.grad_forget_gates{1,i},obj.output{2,i});
                        obj.grad_input{1,i} = obj.W_il.context' * obj.grad_input_gates{1,i} + obj.W_if.context' * obj.grad_forget_gates{1,i} + ...
                            obj.W_ic.context' * obj.grad_cells{1,i} + obj.W_iw.context' * obj.grad_output_gates{1,i};
                    else
                        obj.grad_forget_gates{1,i} = obj.gate_diff_activ(obj.forget_gates{1,i}) .* obj.init_state{1,1} .* obj.grad_states{1,i}; 
                        obj.grad_forget_gates{1,i} = bsxfun(@times,obj.grad_forget_gates{1,i},obj.output{2,i});
                        obj.grad_input{1,i} = obj.W_il.context' * obj.grad_input_gates{1,i} + obj.W_if.context' * obj.grad_forget_gates{1,i} + ...
                            obj.W_ic.context' * obj.grad_cells{1,i} + obj.W_iw.context' * obj.grad_output_gates{1,i};
                    end
                end
                obj.grad_input{1,i} = bsxfun(@times,obj.grad_input{1,i},obj.output{2,i});
            end
            grad_input = obj.grad_input;
            for i = 1 : obj.length
                obj.grad_B_l.context = obj.grad_B_l.context + sum(obj.grad_input_gates{1,i},2) ./ sum(obj.output{2,i},2);
                obj.grad_B_c.context = obj.grad_B_c.context + sum(obj.grad_cells{1,i},2) ./ sum(obj.output{2,i},2);
                obj.grad_B_w.context = obj.grad_B_w.context + sum(obj.grad_output_gates{1,i},2) ./ sum(obj.output{2,i},2);
                obj.grad_B_f.context = obj.grad_B_f.context + sum(obj.grad_forget_gates{1,i},2) ./ sum(obj.output{2,i},2);
                obj.grad_W_il.context = obj.grad_W_il.context + obj.grad_input_gates{1,i} * (obj.input{1,i})' ./ sum(obj.output{2,i},2);
                obj.grad_W_ic.context = obj.grad_W_ic.context + obj.grad_cells{1,i} * (obj.input{1,i})' ./ sum(obj.output{2,i},2);
                obj.grad_W_iw.context = obj.grad_W_iw.context + obj.grad_output_gates{1,i} * (obj.input{1,i})' ./ sum(obj.output{2,i},2);
                obj.grad_W_cw.context = obj.grad_W_cw.context + obj.grad_output_gates{1,i} * (obj.states{1,i})' ./ sum(obj.output{2,i},2);
                obj.grad_W_if.context = obj.grad_W_if.context + obj.grad_forget_gates{1,i} * (obj.input{1,i})' ./ sum(obj.output{2,i},2);
                if i > 1
                    obj.grad_W_hl.context = obj.grad_W_hl.context + obj.grad_input_gates{1,i} * (obj.output{1,i - 1})' ./ ...
                        (obj.output{2,i} * obj.output{2,i - 1}');
                    obj.grad_W_cl.context = obj.grad_W_cl.context + obj.grad_input_gates{1,i} * (obj.states{1,i - 1})' ./ ...
                        (obj.output{2,i} * obj.output{2,i - 1}');
                    obj.grad_W_hf.context = obj.grad_W_hf.context + obj.grad_forget_gates{1,i} * (obj.output{1,i - 1})' ./ ...
                        (obj.output{2,i} * obj.output{2,i - 1}');
                    obj.grad_W_cf.context = obj.grad_W_cf.context + obj.grad_forget_gates{1,i} * (obj.states{1,i - 1})' ./ ...
                        (obj.output{2,i} * obj.output{2,i - 1}');
                    obj.grad_W_hc.context = obj.grad_W_hc.context + obj.grad_cells{1,i} * (obj.output{1,i - 1})' ./ ...
                        (obj.output{2,i} * obj.output{2,i - 1}');
                    obj.grad_W_hw.context = obj.grad_W_hw.context + obj.grad_output_gates{1,i} * (obj.output{1,i - 1})' ./ ...
                        (obj.output{2,i} * obj.output{2,i - 1}');
                else
                    obj.grad_W_hl.context = obj.grad_W_hl.context + obj.grad_input_gates{1,i} * (obj.init_output{1,1})' ./ (obj.init_output{2,1} * obj.output{2,1}');
                    obj.grad_W_cl.context = obj.grad_W_cl.context + obj.grad_input_gates{1,i} * (obj.init_state{1,1})' ./ (obj.init_output{2,1} * obj.output{2,1}');
                    obj.grad_W_hf.context = obj.grad_W_hf.context + obj.grad_forget_gates{1,i} * (obj.init_output{1,1})' ./ (obj.init_output{2,1} * obj.output{2,1}');
                    obj.grad_W_cf.context = obj.grad_W_cf.context + obj.grad_forget_gates{1,i} * (obj.init_state{1,1})' ./ (obj.init_output{2,1} * obj.output{2,1}');
                    obj.grad_W_hc.context = obj.grad_W_hc.context + obj.grad_cells{1,i} * (obj.init_output{1,1})' ./ (obj.init_output{2,1} * obj.output{2,1}');
                    obj.grad_W_hw.context = obj.grad_W_hw.context + obj.grad_output_gates{1,i} * (obj.init_output{1,1})' ./ (obj.init_output{2,1} * obj.output{2,1}');
                end
            end
            obj.grad_init_output{1,1} = obj.W_hl.context' * obj.grad_input_gates{1,1} + ...
                 obj.W_hf.context' * obj.grad_forget_gates{1,1} + obj.W_hc.context' * obj.grad_cells{1,1} + ...
                 obj.W_hw.context' * obj.grad_output_gates{1,1};
             obj.grad_init_state{1,1} = obj.W_cl.context' * obj.grad_input_gates{1,1} + ...
                 obj.W_cf.context' * obj.grad_forget_gates{1,1} + obj.forget_gates{1,1} .* obj.grad_states{1,1};
             
            if obj.debug
                display(['LstmLayer | grad_W_il | mean : ',num2str(mean(obj.grad_W_il.context(:))),' | std : ',num2str(std(obj.grad_W_il.context(:)))]);
                display(['LstmLayer | grad_W_hl | mean : ',num2str(mean(obj.grad_W_hl.context(:))),' | std : ',num2str(std(obj.grad_W_hl.context(:)))]);
                display(['LstmLayer | grad_W_cl | mean : ',num2str(mean(obj.grad_W_cl.context(:))),' | std : ',num2str(std(obj.grad_W_cl.context(:)))]);
                display(['LstmLayer | grad_W_cf | mean : ',num2str(mean(obj.grad_W_cf.context(:))),' | std : ',num2str(std(obj.grad_W_cf.context(:)))]);
                display(['LstmLayer | grad_W_hf | mean : ',num2str(mean(obj.grad_W_hf.context(:))),' | std : ',num2str(std(obj.grad_W_hf.context(:)))]);
                display(['LstmLayer | grad_W_if | mean : ',num2str(mean(obj.grad_W_if.context(:))),' | std : ',num2str(std(obj.grad_W_if.context(:)))]);
                display(['LstmLayer | grad_W_hc | mean : ',num2str(mean(obj.grad_W_hc.context(:))),' | std : ',num2str(std(obj.grad_W_hc.context(:)))]);
                display(['LstmLayer | grad_W_ic | mean : ',num2str(mean(obj.grad_W_ic.context(:))),' | std : ',num2str(std(obj.grad_W_ic.context(:)))]);
                display(['LstmLayer | grad_W_cw | mean : ',num2str(mean(obj.grad_W_cw.context(:))),' | std : ',num2str(std(obj.grad_W_cw.context(:)))]);
                display(['LstmLayer | grad_W_hw | mean : ',num2str(mean(obj.grad_W_hw.context(:))),' | std : ',num2str(std(obj.grad_W_hw.context(:)))]);
                display(['LstmLayer | grad_W_iw | mean : ',num2str(mean(obj.grad_W_iw.context(:))),' | std : ',num2str(std(obj.grad_W_iw.context(:)))]);
            end
        end
        
        function initialOption(obj,option)
            initialOption@OperateLayer(obj,option)
            
            if isfield(option,'gate_activation')
                obj.gate_activation = option.gate_activation;
            end
            
            if isfield(option,'gate_diff_activ')
                obj.gate_diff_activ = option.gate_diff_activ;
            end
            
            if isfield(option,'cell_activation')
                obj.cell_activation = option.cell_activation;
            end
            
            if isfield(option,'cell_diff_activ')
                obj.cell_diff_activ = option.cell_diff_activ;
            end
        end
        
        function update(obj,apply,option)
            if nargin <= 2
                option = struct();
            end
            obj.W_il.context = apply(obj.W_il.context,obj.grad_W_il.context,option);
            obj.W_hl.context = apply(obj.W_hl.context,obj.grad_W_hl.context,option);
            obj.W_cl.context = apply(obj.W_cl.context,obj.grad_W_cl.context,option);
            obj.W_cf.context = apply(obj.W_cf.context,obj.grad_W_cf.context,option);
            obj.W_hf.context = apply(obj.W_hf.context,obj.grad_W_hf.context,option);
            obj.W_if.context = apply(obj.W_if.context,obj.grad_W_if.context,option);
            obj.W_hc.context = apply(obj.W_hc.context,obj.grad_W_hc.context,option);
            obj.W_ic.context = apply(obj.W_ic.context,obj.grad_W_ic.context,option);
            obj.W_cw.context = apply(obj.W_cw.context,obj.grad_W_cw.context,option);
            obj.W_hw.context = apply(obj.W_hw.context,obj.grad_W_hw.context,option);
            obj.W_iw.context = apply(obj.W_iw.context,obj.grad_W_iw.context,option);
            obj.B_l.context = apply(obj.B_l.context,obj.grad_B_l.context,option);
            obj.B_f.context = apply(obj.B_f.context,obj.grad_B_f.context,option);
            obj.B_c.context = apply(obj.B_c.context,obj.grad_B_c.context,option);
            obj.B_w.context = apply(obj.B_w.context,obj.grad_B_w.context,option);
            obj.grad_W_il.setZeros();
            obj.grad_W_hl.setZeros();
            obj.grad_W_cl.setZeros();
            obj.grad_W_cf.setZeros();
            obj.grad_W_hf.setZeros();
            obj.grad_W_if.setZeros();
            obj.grad_W_hc.setZeros();
            obj.grad_W_ic.setZeros();
            obj.grad_W_cw.setZeros();
            obj.grad_W_hw.setZeros();
            obj.grad_W_iw.setZeros();
            obj.grad_B_l.setZeros();
            obj.grad_B_f.setZeros();
            obj.grad_B_c.setZeros();
            obj.grad_B_w.setZeros();
        end 
        
        function object = saveObj(obj)
            object = struct();
            if obj.W_il.useGPU
                object.W_il = gather(obj.W_il.context);
                object.W_hl = gather(obj.W_hl.context);
                object.W_cl = gather(obj.W_cl.context);
                object.W_cf = gather(obj.W_cf.context);
                object.W_hf = gather(obj.W_hf.context);
                object.W_if = gather(obj.W_if.context);
                object.W_hc = gather(obj.W_hc.context);
                object.W_ic = gather(obj.W_ic.context);
                object.W_cw = gather(obj.W_cw.context);
                object.W_hw = gather(obj.W_hw.context);
                object.W_iw = gather(obj.W_iw.context);
                object.B_l = gather(obj.B_l.context);
                object.B_f = gather(obj.B_f.context);
                object.B_c = gather(obj.B_c.context);
                object.B_w = gather(obj.B_w.context);
            else
                object.W_il = obj.W_il.context;
                object.W_hl = obj.W_hl.context;
                object.W_cl = obj.W_cl.context;
                object.W_cf = obj.W_cf.context;
                object.W_hf = obj.W_hf.context;
                object.W_if = obj.W_if.context;
                object.W_hc = obj.W_hc.context;
                object.W_ic = obj.W_ic.context;
                object.W_cw = obj.W_cw.context;
                object.W_hw = obj.W_hw.context;
                object.W_iw = obj.W_iw.context;
                object.B_l = obj.B_l.context;
                object.B_f = obj.B_f.context;
                object.B_c = obj.B_c.context;
                object.B_w = obj.B_w.context;
            end
            object.gate_activation = obj.gate_activation;
            object.gate_diff_activ = obj.gate_diff_activ;
            object.cell_activation = obj.cell_activation;
            object.cell_diff_activ = obj.cell_diff_activ;
            object.activation = obj.activation;
            object.diff_activ = obj.diff_activ;
        end
        
        function loadObj(obj,object)
            obj.W_il.context = obj.init.dataConvert(object.W_il);
            obj.W_hl.context = obj.init.dataConvert(object.W_hl);
            obj.W_cl.context = obj.init.dataConvert(object.W_cl);
            obj.W_cf.context = obj.init.dataConvert(object.W_cf);
            obj.W_hf.context = obj.init.dataConvert(object.W_hf);
            obj.W_if.context = obj.init.dataConvert(object.W_if);
            obj.W_hc.context = obj.init.dataConvert(object.W_hc);
            obj.W_ic.context = obj.init.dataConvert(object.W_ic);
            obj.W_cw.context = obj.init.dataConvert(object.W_cw);
            obj.W_hw.context = obj.init.dataConvert(object.W_hw);
            obj.W_iw.context = obj.init.dataConvert(object.W_iw);
            obj.B_l.context = obj.init.dataConvert(object.B_l);
            obj.B_f.context = obj.init.dataConvert(object.B_f);
            obj.B_c.context = obj.init.dataConvert(object.B_c);
            obj.B_w.context = obj.init.dataConvert(object.B_w);
            
            obj.gate_activation = object.gate_activation;
            obj.gate_diff_activ = object.gate_diff_activ;
            obj.cell_activation = object.cell_activation;
            obj.cell_diff_activ = object.cell_diff_activ;
            obj.activation = object.activation;
            obj.diff_activ = object.diff_activ;
        end
        %% the functions below this line are used in the above functions or some functions are just defined for the gradient check;
        function checkGrad(obj)
            % prepare the data 
            seqLen = 20;
            batchSize = 20;
            input = cell([2,seqLen]);
            target = cell([1,seqLen]);
            mask = ones(seqLen,batchSize);
            truncate = randi(seqLen - 1,1,batchSize);
            for i = 1 : batchSize - 1
                mask( 1 : truncate(1,i),i) = 1;
            end
            mask(:,batchSize) = 1;
            for i = 1 : seqLen
                input{2,i} = mask(i,:);
                input{1,i} = bsxfun(@times,randn([obj.input_dim,batchSize]),mask(i,:));
                target{1,i} = bsxfun(@times,randn([obj.hidden_dim,batchSize]),mask(i,:));
            end
            epislon = 10 ^ (-7);
            
            W_il = obj.W_il.context;
            W_hl = obj.W_hl.context;
            W_cl = obj.W_cl.context;
            W_cf = obj.W_cf.context;
            W_hf = obj.W_hf.context;
            W_if = obj.W_if.context;
            W_hc = obj.W_hc.context;
            W_ic = obj.W_ic.context;
            W_cw = obj.W_cw.context;
            W_hw = obj.W_hw.context;
            W_iw = obj.W_iw.context;
            B_l  = obj.B_l.context;
            B_f  = obj.B_f.context;
            B_c  = obj.B_c.context;
            B_w  = obj.B_w.context;
            
            obj.fprop(input,size(input,2));
            grad_output = cell([1,size(obj.output,2)]);
            for i = 1 : size(obj.output,2)
                grad_output{1,i} = bsxfun(@times,2 * (obj.output{1,i} - target{1,i}),obj.output{2,i});
            end
            obj.bprop(grad_output);
            
            grad_W_il = obj.grad_W_il.context;
            numeric_grad_W_il = zeros(size(W_il));
            
            grad_W_hl = obj.grad_W_hl.context;
            numeric_grad_W_hl = zeros(size(W_hl));
            
            grad_W_cl = obj.grad_W_cl.context;
            numeric_grad_W_cl = zeros(size(W_cl));
            
            grad_W_cf = obj.grad_W_cf.context;
            numeric_grad_W_cf = zeros(size(W_cf));
            
            grad_W_hf = obj.grad_W_hf.context;
            numeric_grad_W_hf = zeros(size(W_hf));
            
            grad_W_if = obj.grad_W_if.context;
            numeric_grad_W_if = zeros(size(W_if));
            
            grad_W_hc = obj.grad_W_hc.context;
            numeric_grad_W_hc = zeros(size(W_hc));
            
            grad_W_ic = obj.grad_W_ic.context;
            numeric_grad_W_ic = zeros(size(W_ic));
            
            grad_W_cw = obj.grad_W_cw.context;
            numeric_grad_W_cw = zeros(size(W_cw));
            
            grad_W_hw = obj.grad_W_hw.context;
            numeric_grad_W_hw = zeros(size(W_hw));
            
            grad_W_iw = obj.grad_W_iw.context;
            numeric_grad_W_iw = zeros(size(W_iw));
            
            grad_B_l = obj.grad_B_l.context;
            numeric_grad_B_l = zeros(size(B_l));
            
            grad_B_f = obj.grad_B_f.context;
            numeric_grad_B_f = zeros(size(B_f));
            
            grad_B_c = obj.grad_B_c.context;
            numeric_grad_B_c = zeros(size(B_c));
            
            grad_B_w = obj.grad_B_w.context;
            numeric_grad_B_w = zeros(size(B_w));
            
            init_output = obj.init_output{1,1};
            grad_init_output = obj.grad_init_output{1,1};
            
            init_state = obj.init_state{1,1};
            grad_init_state = obj.grad_init_state{1,1};
            
            grad_input = obj.grad_input;
            numeric_grad_input = cell(size(grad_input));
            for i = 1 : size(numeric_grad_input,2)
                numeric_grad_input{1,i} = zeros(size(grad_input{1,i}));
            end
            %% the W_il parameter check
            for n = 1 : size(W_il,1)
                for m = 1 : size(W_il,2)
                    obj.W_il.context = W_il;
                    obj.W_il.context(n,m) = obj.W_il.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_il.context = W_il;
                    obj.W_il.context(n,m) = obj.W_il.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_il(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_il(:) - grad_W_il(:)) ./ norm(numeric_grad_W_il(:) + grad_W_il(:));
            if obj.debug
                disp([numeric_grad_W_il(:),obj.grad_W_il.context(:)]);
            end
            disp(['the W_il parameter check is ' , num2str(norm_diff)])
            %% the W_hl parameter check
            for n = 1 : size(W_hl,1)
                for m = 1 : size(W_hl,2)
                    obj.W_hl.context = W_hl;
                    obj.W_hl.context(n,m) = obj.W_hl.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_hl.context = W_hl;
                    obj.W_hl.context(n,m) = obj.W_hl.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_hl(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_hl(:) - grad_W_hl(:)) ./ norm(numeric_grad_W_hl(:) + grad_W_hl(:));
            if obj.debug
                disp([numeric_grad_W_hl(:),obj.grad_W_hl.context(:)]);
            end
            disp(['the W_hl parameter check is ' , num2str(norm_diff)])
            %% the W_cl parameter check
            for n = 1 : size(W_cl,1)
                for m = 1 : size(W_cl,2)
                    obj.W_cl.context = W_cl;
                    obj.W_cl.context(n,m) = obj.W_cl.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_cl.context = W_cl;
                    obj.W_cl.context(n,m) = obj.W_cl.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_cl(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_cl(:) - grad_W_cl(:)) ./ norm(numeric_grad_W_cl(:) + grad_W_cl(:));
            if obj.debug
                disp([numeric_grad_W_cl(:),obj.grad_W_cl.context(:)]);
            end
            disp(['the W_cl parameter check is ' , num2str(norm_diff)])
            %% the W_cf parameter check
            for n = 1 : size(W_cf,1)
                for m = 1 : size(W_cf,2)
                    obj.W_cf.context = W_cf;
                    obj.W_cf.context(n,m) = obj.W_cf.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_cf.context = W_cf;
                    obj.W_cf.context(n,m) = obj.W_cf.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_cf(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_cf(:) - grad_W_cf(:)) ./ norm(numeric_grad_W_cf(:) + grad_W_cf(:));
            if obj.debug
                disp([numeric_grad_W_cf(:),obj.grad_W_cf.context(:)]);
            end
            disp(['the W_cf parameter check is ' , num2str(norm_diff)])
            %% the W_hf parameter check
            for n = 1 : size(W_hf,1)
                for m = 1 : size(W_hf,2)
                    obj.W_hf.context = W_hf;
                    obj.W_hf.context(n,m) = obj.W_hf.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_hf.context = W_hf;
                    obj.W_hf.context(n,m) = obj.W_hf.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_hf(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_hf(:) - grad_W_hf(:)) ./ norm(numeric_grad_W_hf(:) + grad_W_hf(:));
            if obj.debug
                disp([numeric_grad_W_hf(:),obj.grad_W_hf.context(:)]);
            end
            disp(['the W_hf parameter check is ' , num2str(norm_diff)])
            %% the W_if parameter check
            for n = 1 : size(W_if,1)
                for m = 1 : size(W_if,2)
                    obj.W_if.context = W_if;
                    obj.W_if.context(n,m) = obj.W_if.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_if.context = W_if;
                    obj.W_if.context(n,m) = obj.W_if.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_if(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_if(:) - grad_W_if(:)) ./ norm(numeric_grad_W_if(:) + grad_W_if(:));
            if obj.debug
                disp([numeric_grad_W_if(:),obj.grad_W_if.context(:)]);
            end
            disp(['the W_if parameter check is ' , num2str(norm_diff)])
            %% the W_hc parameter check
            for n = 1 : size(W_hc,1)
                for m = 1 : size(W_hc,2)
                    obj.W_hc.context = W_hc;
                    obj.W_hc.context(n,m) = obj.W_hc.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_hc.context = W_hc;
                    obj.W_hc.context(n,m) = obj.W_hc.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_hc(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_hc(:) - grad_W_hc(:)) ./ norm(numeric_grad_W_hc(:) + grad_W_hc(:));
            if obj.debug
                disp([numeric_grad_W_hc(:),obj.grad_W_hc.context(:)]);
            end
            disp(['the W_hc parameter check is ' , num2str(norm_diff)])
            %% the W_ic parameter check
            for n = 1 : size(W_ic,1)
                for m = 1 : size(W_ic,2)
                    obj.W_ic.context = W_ic;
                    obj.W_ic.context(n,m) = obj.W_ic.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_ic.context = W_ic;
                    obj.W_ic.context(n,m) = obj.W_ic.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_ic(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_ic(:) - grad_W_ic(:)) ./ norm(numeric_grad_W_ic(:) + grad_W_ic(:));
            if obj.debug
                disp([numeric_grad_W_ic(:),obj.grad_W_ic.context(:)]);
            end
            disp(['the W_ic parameter check is ' , num2str(norm_diff)])
            %% the W_cw parameter check
            for n = 1 : size(W_cw,1)
                for m = 1 : size(W_cw,2)
                    obj.W_cw.context = W_cw;
                    obj.W_cw.context(n,m) = obj.W_cw.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_cw.context = W_cw;
                    obj.W_cw.context(n,m) = obj.W_cw.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_cw(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_cw(:) - grad_W_cw(:)) ./ norm(numeric_grad_W_cw(:) + grad_W_cw(:));
            if obj.debug
                disp([numeric_grad_W_cw(:),obj.grad_W_cw.context(:)]);
            end
            disp(['the W_cw parameter check is ' , num2str(norm_diff)])
            %% the W_hw parameter check
            for n = 1 : size(W_hw,1)
                for m = 1 : size(W_hw,2)
                    obj.W_hw.context = W_hw;
                    obj.W_hw.context(n,m) = obj.W_hw.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_hw.context = W_hw;
                    obj.W_hw.context(n,m) = obj.W_hw.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_hw(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_hw(:) - grad_W_hw(:)) ./ norm(numeric_grad_W_hw(:) + grad_W_hw(:));
            if obj.debug
                disp([numeric_grad_W_hw(:),obj.grad_W_hw.context(:)]);
            end
            disp(['the W_hw parameter check is ' , num2str(norm_diff)])
            %% the W_iw parameter check
            for n = 1 : size(W_iw,1)
                for m = 1 : size(W_iw,2)
                    obj.W_iw.context = W_iw;
                    obj.W_iw.context(n,m) = obj.W_iw.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.W_iw.context = W_iw;
                    obj.W_iw.context(n,m) = obj.W_iw.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_W_iw(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_W_iw(:) - grad_W_iw(:)) ./ norm(numeric_grad_W_iw(:) + grad_W_iw(:));
            if obj.debug
                disp([numeric_grad_W_iw(:),obj.grad_W_iw.context(:)]);
            end
            disp(['the W_iw parameter check is ' , num2str(norm_diff)])
            %% the B_l parameter check
            for n = 1 : size(B_l,1)
                for m = 1 : size(B_l,2)
                    obj.B_l.context = B_l;
                    obj.B_l.context(n,m) = obj.B_l.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.B_l.context = B_l;
                    obj.B_l.context(n,m) = obj.B_l.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_B_l(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_B_l(:) - grad_B_l(:)) ./ norm(numeric_grad_B_l(:) + grad_B_l(:));
            if obj.debug
                disp([numeric_grad_B_l(:),obj.grad_B_l.context(:)]);
            end
            disp(['the B_l parameter check is ' , num2str(norm_diff)])
            %% the B_f parameter check
            for n = 1 : size(B_f,1)
                for m = 1 : size(B_f,2)
                    obj.B_f.context = B_f;
                    obj.B_f.context(n,m) = obj.B_f.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.B_f.context = B_f;
                    obj.B_f.context(n,m) = obj.B_f.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_B_f(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_B_f(:) - grad_B_f(:)) ./ norm(numeric_grad_B_f(:) + grad_B_f(:));
            if obj.debug
                disp([numeric_grad_B_f(:),obj.grad_B_f.context(:)]);
            end
            disp(['the B_f parameter check is ' , num2str(norm_diff)])
            %% the B_c parameter check
            for n = 1 : size(B_c,1)
                for m = 1 : size(B_c,2)
                    obj.B_c.context = B_c;
                    obj.B_c.context(n,m) = obj.B_c.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.B_c.context = B_c;
                    obj.B_c.context(n,m) = obj.B_c.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_B_c(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_B_c(:) - grad_B_c(:)) ./ norm(numeric_grad_B_c(:) + grad_B_c(:));
            if obj.debug
                disp([numeric_grad_B_c(:),obj.grad_B_c.context(:)]);
            end
            disp(['the B_c parameter check is ' , num2str(norm_diff)])
            %% the B_w parameter check
            for n = 1 : size(B_w,1)
                for m = 1 : size(B_w,2)
                    obj.B_w.context = B_w;
                    obj.B_w.context(n,m) = obj.B_w.context(n,m) + epislon;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    obj.B_w.context = B_w;
                    obj.B_w.context(n,m) = obj.B_w.context(n,m) - epislon;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_B_w(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_B_w(:) - grad_B_w(:)) ./ norm(numeric_grad_B_w(:) + grad_B_w(:));
            if obj.debug
                disp([numeric_grad_B_w(:),obj.grad_B_w.context(:)]);
            end
            disp(['the B_w parameter check is ' , num2str(norm_diff)])
            
            %%  the init_output parameter check
            for n = 1 : size(init_output,1)
                for m = 1 : size(init_output,2)
                    temp_init_output = init_output;
                    temp_init_output(n,m) = temp_init_output(n,m) + epislon;
                    obj.init_output{1,1} = temp_init_output;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    temp_init_output = init_output;
                    temp_init_output(n,m) = temp_init_output(n,m) - epislon;
                    obj.init_output{1,1} = temp_init_output;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_init_output(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_init_output(:) - grad_init_output(:)) ./ norm(numeric_grad_init_output(:) + grad_init_output(:));
            if obj.debug
                disp([numeric_grad_init_output(:),grad_init_output(:)]);
            end
            disp(['the init_output parameter check is ' , num2str(norm_diff)])
            
            %%  the init_state parameter check
            for n = 1 : size(init_state,1)
                for m = 1 : size(init_state,2)
                    temp_init_state = init_state;
                    temp_init_state(n,m) = temp_init_state(n,m) + epislon;
                    obj.init_state{1,1} = temp_init_state;
                    obj.fprop(input,size(input,2));
                    cost_1 = getCost(target,obj.output);
                    
                    temp_init_state = init_state;
                    temp_init_state(n,m) = temp_init_state(n,m) - epislon;
                    obj.init_state{1,1} = temp_init_state;
                    obj.fprop(input,size(input,2));
                    cost_2 = getCost(target,obj.output);
                    
                    numeric_grad_init_state(n,m) = (cost_1 - cost_2) ./ (2 * epislon);
                end
            end
            norm_diff = norm(numeric_grad_init_state(:) - grad_init_state(:)) ./ norm(numeric_grad_init_state(:) + grad_init_state(:));
            if obj.debug
                disp([numeric_grad_init_state(:),grad_init_state(:)]);
            end
            disp(['the init_state parameter check is ' , num2str(norm_diff)])
            %% check the gradient of input data
            for t = 1 : seqLen
                temp = input{1,t};
                for i = 1 : size(temp,1)
                    for j = 1 : size(temp,2)
                        if input{2,t}(1,j) == 0
                            continue;
                        end
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
                norm_diff = norm(numeric_grad_input{1,t}(:) - grad_input{1,t}(:)) ./ norm(numeric_grad_input{1,t}(:) + grad_input{1,t}(:));
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
        temp = bsxfun(@times,temp,output{2,m});
        cost = cost + sum(temp(:)) ./ sum(output{2,m},2);
    end
end