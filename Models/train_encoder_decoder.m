clc;
clear;
close all;
% dbstop if error;
setup;
%% setting definition

gpuDevice(4);

useGPU = true;
dataType = 'single';
backward = true;
debug = false;
batchSize = 100;
permute = true;
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

%% train
MaxIter = 100000;
history_cost = zeros(1,MaxIter);
history_accuracy = zeros(1,MaxIter);
context = cell([1,1]);

front_add = cell([1,1]);
front_add{1,1} = convert.dataConvert(3 * ones(1,batchSize));
backe_add = cell([1,1]);
encode_grad_output = cell(1);
convert.setDataSize([encoder.hidden_dim,batchSize]);
convert.setZeros();
layers = cell(1);
for i = 1 : MaxIter
    tic;
    %% forward
    temp = post.base_point + post.batchSize;
    if (temp > size(post.txt.data,1)) || isempty(post.permutation)
        post.permutation = randperm(size(post.txt.data,1));
        cmnt.permutation = post.permutation;
    end
    % encode
    display(' == == == == == == train == == == == == == ');
    input = post.fprop(struct('reverse',true));
    display(['encode input start : ',num2str(post.used_index(1,1)),' end : ',num2str(post.used_index(1,end))]);
    display(['the length of encode input : ',num2str(size(input,2))]);
    embedd1.fprop(input,size(input,2));
    encoder.fprop(embedd1.output,size(input,2));
    display(['the length of encode output : ',num2str(size(encoder.output,2))]);

    % decode
    decoder.init_output{1,1} = encoder.output{1,size(input,2)};
    decoder.init_state{1,1}  = encoder.states{1,size(input,2)};
    decode_raw = cmnt.fprop();
    display(['decode input start : ',num2str(cmnt.used_index(1,1)),' end : ',num2str(cmnt.used_index(1,end))]);
    display(['the length of decode input : ',num2str(size(decode_raw,2) + 1)]);
    for j = 2 : size(decode_raw,2) + 1
        front_add{1,j} = decode_raw{1,j - 1};
    end
    for j = 1 : size(decode_raw,2)
        backe_add{1,j} = decode_raw{1,j};
    end
    backe_add{1,size(decode_raw,2) + 1} = convert.dataConvert(ones([1,batchSize]));
    embedd1.fprop(front_add,size(front_add,2));
    decoder.fprop(embedd1.output,size(front_add,2));
    loss.fprop(decoder.output,size(front_add,2));
    display(['the length of decode output : ',num2str(size(loss.output,2))]);
    history_cost(1,i) = gather(loss.getCost(backe_add));
    history_accuracy(1,i) = gather(loss.getAccuracy(backe_add));
    % show cost
    display(['iteration : ',num2str(i)]);
    display(['cost      : ',num2str(history_cost(1,i))]);
    display(['accuracy  : ',num2str(history_accuracy(1,i))]);

    %% backward
    % decode
    loss.bprop(backe_add);
    decoder.bprop(loss.grad_input);
    for j = 1 : size(input,2)
        encode_grad_output{1,j} = convert.context;
    end
    encode_grad_output{1,size(input,2)} = decoder.grad_init_output{1,1};
    encoder.grad_output_state{1,1} = decoder.grad_init_state{1,1};

    encoder.bprop(encode_grad_output);
    embedd1.bprop(encoder.grad_input,size(encoder.grad_input,2));
    embedd1.bprop(decoder.grad_input,size(decoder.grad_input,2));
    embedd1.update(@SGD);
    encoder.update(@SGD);
    decoder.update(@SGD);
    loss.update(@SGD);
    toc;

    if mod(i,100) == 1
        display(' ## ## ## ## ## ## ## ## ');
        layers{1,1} = embedd1.saveObj();
        layers{2,1} = 'EmbeddingLayer';
        layers{1,2} = encoder.saveObj();
        layers{2,2} = 'LstmLayer';
        layers{1,3} = decoder.saveObj();
        layers{2,3} = 'LstmLayer';
        layers{1,4} = loss.saveObj();
        layers{2,4} = 'SoftmaxLayer';
        save('encoder_decoder.mat','layers');
        display(' ## ## ## ## ## ## ## ## ');
    end
end








