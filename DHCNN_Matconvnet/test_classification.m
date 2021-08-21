function [OA] = test_classification(net, test_data, test_L, no_classes)
batchsize = 128;
predict_label = [];
res = [];

for j = 0:ceil(size(test_data,4)/batchsize)-1
    im = test_data(:,:,:,(1+j*batchsize):min((j+1)*batchsize,size(test_data,4))) ;
    im_ = gpuArray(im) ;
    Lab = test_L((1+j*batchsize):min((j+1)*batchsize,size(test_data,4)))';
    net.layers{end}.class = Lab';
    dzdy = 0;
    % run the CNN
        res = vl_simplenn(net, im_) ;
    output_values = vl_nnsoftmax(res(end).x) ;
    output_values = squeeze(gather(output_values))';
%     output_values = squeeze(gather(res(end).x))' ;
    [~,location] = sort(output_values, 2, 'descend');
    predict_label_temp = location(:,1);
    predict_label = [predict_label ; predict_label_temp];
end

[OA,Kappa,AA,CA, cfm] = calcError(test_L-1,predict_label-1,[1:no_classes]);

end