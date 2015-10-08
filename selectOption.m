function newOption = selectOption(option,varargin)
newOption = struct();
for i = 1 : size(varargin,2)
    if isfield(option,varargin{i})
        eval(['newOption.',varargin{i},' = option.',varargin{i},';']);
    end
end
end