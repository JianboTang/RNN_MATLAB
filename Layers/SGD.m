function W = SGD(W,grad_W,option)
if nargin <= 2
    option = struct();
end
if ~isfield(option,'learningRate')
    option.learningRate = 0.005;
end
W = W - option.learningRate .* grad_W;
end