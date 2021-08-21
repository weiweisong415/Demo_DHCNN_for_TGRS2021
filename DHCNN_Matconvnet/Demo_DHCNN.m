%%%%   main function to achieve remote sensing image retrieval and classification  %%%%%%
% This code implementation is refererd to DPSH( https://cs.nju.edu.cn/lwj/)
% author: Weiwei Song
% If you have any questions, please contact me: weiweisong415@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
run(fullfile(fileparts(mfilename('fullpath')),...
    'matconvnet-1.0-beta25', 'matlab', 'vl_setupnn.m')) ;
dataset_name = 'UCM21'; % CIFAR10; UCM21; AID30;WHU19;NWPU45; PatternNet
img_type =  '*.tif';% UCM21='*.tif', AID30='*.jpg';WHU19='*.jpg';NWPU45='*.jpg';
net_name = 'imagenet-vgg-f.mat';  % imagenet-vgg-verydeep-16.mat   imagenet-vgg-f.mat  imagenet-matconvnet-alex

switch(dataset_name)
    case 'CIFAR10'
        no_classes = 10;
        num_perclass = ones(1,no_classes)*6000;
        train_ratio = 5/6;
        train_perclass = ceil(num_perclass*train_ratio);
    case 'UCM21'
        no_classes = 21;
        num_perclass = ones(1,no_classes)*100;
        train_ratio = 0.8;
        train_perclass = ceil(num_perclass*train_ratio);
    case 'AID30'
        no_classes = 30;
        num_perclass = [360,310,220,400,360,260,240,350,410,300,...
            370,250,390,280,290,340,350,390,370,420,...
            380,260,290,410,300,300,330,290,360,420] ;
        train_ratio = 0.5;
        train_perclass = ceil(num_perclass*train_ratio);
    case 'NWPU45'
        no_classes = 45;
        num_perclass = ones(1,no_classes)*700;
    case 'PatternNet'
        no_classes = 38;
        num_perclass = ones(1,no_classes)*800;
        train_ratio = 0.8;
        train_perclass = ceil(num_perclass*train_ratio);
end
% data_prepare_cifar10;
% load('data/CIFAR-10/cifar-10.mat') ;
img_dir = strcat('data/',dataset_name);

model_dir = strcat('models/',net_name);
codelens = 64;  %  length of hash code
maxIter = 100;  % number of iterations
b_s = 10;  % batch_size
beta = 25;  % appreciation parameter£¬ 25 for UCMD21 and AID30,  10 for WHU19
eta = 0.2;   % 0.1   blanced paramter
num_neib = 15; % number of returned neighbors
pos = [1:10:40 50:50:1000]; % for UCMD and AID :The number of retrieved samples: Recall-The number of retrieved samples curve
% pos = [1:20:500];  % for WHU-RS
% num_query = 10; % number of test query images for show;
[data, label] = data_prepare(img_dir, dataset_name, img_type);
for kk=1:length(eta)
    for ii=1:1
        net = load(model_dir);
        load(strcat(dataset_name, '.mat'));
        [train_data, train_L, test_data, test_L, query_data, query_L] = generating_samples(data, label, no_classes, train_perclass);
        % load('mat_file/UCM21/DATA.mat');
        %% initialization
        lr = logspace(-2,-5,maxIter) ; %generate #maxIter of points between 10^(-2) ~ 10^(-6)
        net = net_structure (net, codelens, no_classes);
        U = zeros(size(train_data,4),codelens);    % hash-like matric
        B = zeros(size(train_data,4),codelens);    % hash matric
        id = randperm(size(train_data,4));
        train_data = train_data(:,:,:,id);
        train_L = train_L(id);     
        
        %% start network train  %%
        tic 
        for iter = 1: maxIter
            [net,U,B] = train(train_data,train_L,U,B,net,iter,lr(iter),eta(kk), no_classes, beta, b_s) ;
        end
        toc
        
        %% compute the metric based on query samples  %%
        [map{kk}(ii),OA{kk}(ii),recall{kk}{ii},precision{kk}{ii}, rec{kk}{ii}, pre{kk}{ii}] = test_retrieval(net, train_L, test_L,train_data, test_data, no_classes, pos )
        %save ('results/DHCNN-UCM21.mat', 'map','OA', 'recall','precision','rec','pre');
        % Query_OA = return_retrieval(net, train_L, query_L,train_data, query_data, num_neib, no_classes )
        clear net data label test_data test_L train_data train_L B U
    end
end