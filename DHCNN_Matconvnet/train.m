function [net, U, B] = train (X1, L1, U, B, net, iter ,lr,eta, no_classes, beta,batchsize)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X1: data, width*heigth*3*num_samples; 
% L1: label, num_samples*1
% U,B: hash code matrix
% eta, beta: parameters in loss function
% iter, lr, batchsize: training parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    num =0;
    N = size(X1,4) ;
    index = randperm(N) ;
    p = 32;  % similarity weight : DHNNL2=32  DPSH=2;
    L1= single(L1);
    for j = 0:ceil(N/batchsize)-1
        batch_time=tic ;
        
        %% random select a minibatch
        ix = index((1+j*batchsize):min((j+1)*batchsize,N)) ;
        S = calcNeighbor (L1, ix, 1:N) ;
        Lab = L1(ix)';
        num = num+length(ix);
        %% load and preprocess an image
       im = X1(:,:,:,ix) ;
       im = single(im) ; % note: 0-255 range
       im = imresize(im, net.meta.normalization.imageSize(1:2)) ;
       im = im - repmat(net.meta.normalization.averageImage,1,1,1,size(im,4)) ;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%
       im = gpuArray(im) ;
%%%%%%%%%%%%%%%%%%   run the network    %%%%%%%%%%%%%%%%%
       net.layers{end}.class = Lab;
       c_lab = 1:no_classes;
       Dp =   repmat(c_lab,length(ix),1) - repmat(Lab',1,length(c_lab));
       lab_gt = Dp == 0;
       res = vl_simplenn(net, im) ;
%%%%%%%%%%%%%%%%%%%  define loss 1 %%%%%%%%%%%%%%       
        dNdU = [];
        U0 = squeeze(gather(res(end-1).x))' ;  %% motified
        U(ix,:) = U0 ;
        B(ix,:) = sign(U0) ;
        T = U0 * U' / p ;
        L1_1 = -sum(sum((S).*T-log(1+exp(T))));
        L1_2 = beta*sum(sum((sign(U0)-U0).^2));
        loss1 = (L1_1+L1_2)/(length(ix)*N);
        A = 1 ./ (1 + exp(-T)) ; 
        dJdU1 =  ((A - S) * U + 2*beta*(U0-sign(U0)));
        dzdy1 = gpuArray(reshape(dJdU1',[1,1,size(dJdU1',1),size(dJdU1',2)])) ;
        res1 = vl_simplenn_new(net, im, dzdy1) ;
        
 %%%%%%%%%%%%%%%%    define loss 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%     
         OUT = vl_nnsoftmax(res(end).x) ;    
         OUT1 = squeeze(gather(OUT))';
         loss2 =  -sum(sum(lab_gt.*log(OUT1)))/length(ix);
         dJdU2 = OUT1 - lab_gt;
         dzdy2 = gpuArray(reshape(dJdU2',[1,1,size(dJdU2',1),size(dJdU2',2)])) ;
         res2 = vl_simplenn(net, im, dzdy2) ;

%%%%%%%%%%%%%%%%%%  update the network   %%%%%%%%%%%%%%%%%%%%%% 
        net = update_net_new(net , res1, res2, lr, length(ix), N, eta) ;  
%         net = update_cifar10(net , res1, lr, N);
        batch_time = toc(batch_time) ;
%         fprintf(' iter %d  batch %d/%d (%.1f images/s) ,lr is %d\n, loss %d, top1err %d', iter, j+1,ceil(size(X1,4)/batchsize), batchsize/ batch_time,lr, stats.objective,stats.top1err) ;
         fprintf(' iter %d  batch %d/%d (%.1f images/s) ,lr is %d\n, loss1 %d, loss2 %d', iter, j+1,ceil(size(X1,4)/batchsize), batchsize/ batch_time,lr, loss1, loss2) ;
    end
end

