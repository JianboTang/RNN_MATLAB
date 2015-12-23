classdef textData < handle
    properties
        dictionary = '../Data/Text/dictionary'
        txtfile = '../Data/Text/test.txt'
        data = [];
        index = [];
    end
    methods
        function td = textData(txtfile,dictionary)
            if nargin == 0
                td.data = textread(td.txtfile);
                td.index = textread(td.dictionary,'%s');
            else if nargin >= 1
                    td.data = textread(txtfile);
                    td.txtfile = txtfile;
                    if nargin >= 2
                        td.dictionary = dictionary;
                        td.index = textread(td.dictionary,'%s');
                    end
                end
            end
        end
    end
end
% function readTxtData()
% end