classdef textData < handle
    properties
        dictionary = '../Data/Text/dictionary'
        txtfile = '../Data/Text/test.txt'
        data = [];
        index = [];
    end
    methods
        function obj = textData(txtfile,dictionary)
            if nargin == 0
                if exist(obj.txtfile,'file')
                    obj.data = textread(obj.txtfile);
                end
                
                if exist(obj.dictionary,'file')
                    obj.index = textread(obj.dictionary,'%s');
                end
            else if nargin >= 1
                    if exist(txtfile,'file')
                        obj.data = textread(txtfile);
                        obj.txtfile = txtfile;
                    end
                    if nargin >= 2
                        if exist(dictionary,'file')
                            obj.dictionary = dictionary;
                            obj.index = textread(obj.dictionary,'%s');
                        end
                    end
                end
            end
        end
    end
end