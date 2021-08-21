function [train_data, train_L, test_data, test_L, query_data, query_L] = generating_samples(data, label, no_classes, train_perclass)
train_data = [];
test_data = [];
train_L = [];
test_L = [];
query_data = [];
query_L =[];
train_index = [];
test_index = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% num_query = 1000;
% n = size(data,4);
% index = randperm(n);
% test_index = index(1:num_query);
% train_index = index(num_query+1:end);
% train_data = data(:,:,:,train_index);
% train_label = label(train_index,:);
% test_data = data(:,:,:,test_index);
% test_label = label(test_index,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ii = 1 : no_classes
    index = find(label==ii);
    data_temp = data(:,:,:,index);
    rand_number = randperm(length(index));
    train_index_temp = rand_number(1:train_perclass(ii));
    train_index = [train_index (ii-1)*length(index)+train_index_temp];
    train_label_temp = ones(1, length(train_index_temp))*ii;
    train_L = [train_L train_label_temp];
    train_data_temp = data_temp(:,:,:,train_index_temp);
%     train_data  = [train_data train_data_temp];
    train_data = cat(4, train_data, train_data_temp);
    test_index_temp = rand_number(train_perclass(ii)+1:end);
    test_index = [test_index (ii-1)*length(index)+test_index_temp];
    test_label_temp = ones(1, length(test_index_temp))*ii;
    test_L = [test_L test_label_temp];
    test_data_temp = data_temp(:,:,:,test_index_temp);
%     test_data = [test_data test_data_temp];
    test_data = cat(4, test_data, test_data_temp);
end
    train_L = train_L';
    test_L = test_L';
    
    id = randperm(size(test_data,4));
    DATA= test_data(:,:,:,id);
    LABEL = test_L(id);
    for jj=1:no_classes
        index=find(LABEL==jj);
        query_data = cat(4, query_data, DATA(:,:,:,index(1)));
        query_L = [query_L; jj];
    end
%     query_id = id(1:num_query);
%     query_data = test_data(:,:,:,query_id);
%     query_label = test_label(query_id);
% save('mat_file/UCM21/UCM21-DATA.mat', 'train_data', 'train_L', 'test_data', 'test_L', 'query_data', 'query_L', 'test_index', 'train_index');
% save('mat_file/AID31/DATA.mat', 'train_data', 'train_L', 'test_data', 'test_L', 'query_data', 'query_L');
end