clc;
clear;
close all;
% dbstop if error;
addpath(genpath('../../toolbox'));
%% setting definition

gpuDevice(3)

useGPU = true;
dataType = 'single';
backward = true;
debug = false;
batchSize = 1;
beamSearch = 10;
permute = false;
convert = Data(struct('useGPU',useGPU,'dataType',dataType));

%% layer definition
post = InputLayer(struct('batchSize',batchSize,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug,'permute',permute));
post.getFile('../data/used/post_index.txt','../data/used/dictionary.txt');

cmnt = InputLayer(struct('batchSize',batchSize,'useGPU',useGPU,'dataType',dataType,'backward',backward,'debug',debug,'permute',permute));
cmnt.getFile('../data/used/cmnt_index.txt','../data/used/dictionary.txt');

embedd1 = EmbeddingLayer(struct('hidden_dim',620,'input_dim',post.vocabSize,'useGPU',useGPU,'dataType',dataType, ...
    'backward',backward,'debug',debug));
encoder = LstmLayer(struct('hidden_dim',500,'input_dim',620,'useGPU',useGPU,'dataType',dataType, ...
    'backward',backward,'debug',debug));
decoder = LstmLayer(struct('hidden_dim',500,'input_dim',620,'useGPU',useGPU,'dataType',dataType, ...
    'backward',backward,'debug',debug));
loss = SoftmaxLayer(struct('hidden_dim',post.vocabSize,'input_dim',500,'useGPU',useGPU,'dataType',dataType, ...
    'backward',backward,'debug',debug));

if exist('encoder_decoder.mat','file')
    load('encoder_decoder.mat');
    embedd1.loadObj(layers{1,1});
    encoder.loadObj(layers{1,2});
    decoder.loadObj(layers{1,3});
    loss.loadObj(layers{1,4});
end
%% test
MaxIter = size(post.txt.data,1);
history_cost = zeros(1,MaxIter);
context = cell([1,1]);

front_add = cell([1,1]);
front_add{1,1} = convert.dataConvert(3 * ones(1,beamSearch));
trans_file = fopen('trans.txt','w+');

convert.setDataSize([encoder.hidden_dim,batchSize]);
convert.setZeros();

layers = cell(1);

for i = 1 : MaxIter
    tic;
    %% forward
    % encode
    input = post.fprop(struct('reverse',true));
    display(['encode input start : ',num2str(post.used_index(1,1)),' end : ',num2str(post.used_index(1,end))]);
    embedd1.fprop(input,size(input,2));
    encoder.fprop(embedd1.output,size(input,2));

    % decode
    previous_index = cell(1);
    previous_index{1,1} = zeros(1,beamSearch);
    current_index = cell(1);
    current_index{1,1} = 3 * ones(1,beamSearch);
    sum_prob = cell(1);
    sum_prob{1,1} = zeros(1,beamSearch);
    
    decoder.init_output{1,1} = repmat(encoder.output{1,size(input,2)},[1,beamSearch]);
    decoder.init_state{1,1} = repmat(encoder.states{1,size(input,2)},[1,beamSearch]);
    temp_step = front_add;
    for j = 1 : 40 %2 * size(input,2)
        if j == 1
            temp_step = embedd1.fprop_step(temp_step,j);
            temp_step = decoder.fprop_step(temp_step,j);
            temp_step = loss.fprop_step(temp_step,j);
            total_prob = bsxfun(@plus,gather(temp_step{1,1}),sum_prob{1,end});
            [max_value,max_pos] = sort(total_prob(:));
            sum_prob{1,end + 1} = (max_value(end - beamSearch * beamSearch + 1 : beamSearch : end,1))';
            previous_index{1,end + 1} = (floor((max_pos(end - beamSearch * beamSearch + 1 : beamSearch : end)  - 1) / size(total_prob,1)) + 1)';
            current_index{1,end + 1} = (mod(max_pos(end - beamSearch * beamSearch + 1 : beamSearch : end)  - 1,size(total_prob,1)) + 1)';
            temp_step{1,1} = convert.dataConvert(current_index{1,end});
        else
            temp_step = embedd1.fprop_step(temp_step,j);
            temp_step = decoder.fprop_step(temp_step,j);
            temp_step = loss.fprop_step(temp_step,j);
            total_prob = bsxfun(@plus,gather(temp_step{1,1}),sum_prob{1,end});
            [max_value,max_pos] = sort(total_prob(:));
            sum_prob{1,end + 1} = (max_value(end - beamSearch + 1 : end,1))';
            previous_index{1,end + 1} = (floor((max_pos(end - beamSearch + 1 : end)  - 1) / size(total_prob,1)) + 1)';
            current_index{1,end + 1} = (mod(max_pos(end - beamSearch + 1 : end)  - 1,size(total_prob,1)) + 1)';
            temp_step{1,1} = convert.dataConvert(current_index{1,end});
        end
        end_mark = current_index{1,end};
        %if sum(end_mark == 1,1) >= 0.5 * size(end_mark,2)
        %    break;
        %end
    end
    trans = zeros(beamSearch,size(sum_prob,2));
    temp = sum_prob{1,j} .* (end_mark == 1);
    [~,candidate_index] = sort(temp);
    for m = 1 : size(candidate_index,2)
        index = candidate_index(1,m);
        trans(m,end) = current_index{1,end}(1,index);
        for j = size(sum_prob,2) - 1 : -1 : 1
            trans(m,j) = current_index{1,j}(previous_index{1,j + 1}(index));
            index = previous_index{1,j}(previous_index{1,j + 1}(index));
        end
        display(['index ',num2str(m),'th candidate']);
        display((post.vocab(trans(m,:)))');
        for j = 1 : size(trans,2)
            fprintf(trans_file,post.vocab{trans(m,j)});
        end
        fprintf(trans_file,' | ');
    end
    fprintf(trans_file,'\n');
    toc;
end
