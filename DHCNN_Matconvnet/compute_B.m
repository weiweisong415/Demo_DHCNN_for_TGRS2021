function [B_dataset, B_test, PL_dataset, PL_test]=compute_B (data_set,test_data,net)
    batchsize = 128;
    for j = 0:ceil(size(data_set,4)/batchsize)-1
        im = data_set(:,:,:,(1+j*batchsize):min((j+1)*batchsize,size(data_set,4))) ;
        im = single(im) ; % note: 0-255 range
        im = imresize(im, net.meta.normalization.imageSize(1:2)) ;
        im = im - repmat(net.meta.normalization.averageImage,1,1,1,size(im,4)) ;
        im = gpuArray(im) ;
        % run the CNN
        res = vl_simplenn(net, im) ;
        features = squeeze(gather(res(end-1).x))' ;
        softmaxt_value = squeeze(gather(res(end).x))' ;
        [~, index] = sort(softmaxt_value, 2, 'descend');
        U_train((1+j*batchsize):min((j+1)*batchsize,size(data_set,4)),:) = features ;
        PL_dataset((1+j*batchsize):min((j+1)*batchsize,size(data_set,4)),:) = index(:,1);
    end
    for j = 0:ceil(size(test_data,4)/batchsize)-1
        im = test_data(:,:,:,(1+j*batchsize):min((j+1)*batchsize,size(test_data,4))) ;
        im = single(im) ; % note: 0-255 range
        im = imresize(im, net.meta.normalization.imageSize(1:2)) ;
        im = im - repmat(net.meta.normalization.averageImage,1,1,1,size(im,4)) ;
        im = gpuArray(im) ;
%         % run the CNN
        res = vl_simplenn(net, im) ;
        features = squeeze(gather(res(end-1).x))' ;
        softmaxt_value = squeeze(gather(res(end).x))' ;
        [~, index] = sort(softmaxt_value, 2, 'descend');
        U_test((1+j*batchsize):min((j+1)*batchsize,size(test_data,4)),:) = features ;
        PL_test((1+j*batchsize):min((j+1)*batchsize,size(test_data,4)),:) = index(:,1);
    end
    B_dataset = U_train > 0 ;
    B_test = U_test > 0 ;
end