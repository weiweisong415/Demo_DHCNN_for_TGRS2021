function net = net_structure (net, codelens, no_classes)
    num = numel(net.layers)-2;
    net.layers = net.layers(1:num);
    n = numel(net.layers) ;
    for i=1:n
        if ~isempty(net.layers{i}.weights)
            net.layers{i}.weights{1} = gpuArray(net.layers{i}.weights{1}) ;
            net.layers{i}.weights{2} = gpuArray(net.layers{i}.weights{2}) ;
        end
    end
    net.layers{n+1}.pad = [0,0,0,0];
    net.layers{n+1}.stride = [1,1];
    net.layers{n+1}.type = 'conv';
    net.layers{n+1}.name = 'hash';
    net.layers{n+1}.weights{1} = gpuArray(0.01*randn(1,1,4096,codelens,'single'));
    net.layers{n+1}.weights{2} = gpuArray(0.01*randn(1,codelens,'single'));
    net.layers{n+1}.opts = {};
    net.layers{n+1}.dilate = 1;
    
    net.layers{n+2}.pad = [0,0,0,0];
    net.layers{n+2}.stride = [1,1];
    net.layers{n+2}.type = 'conv';
    net.layers{n+2}.name = 'classification';
    net.layers{n+2}.weights{1} = gpuArray(0.01*randn(1,1,codelens,no_classes, 'single'));
    net.layers{n+2}.weights{2} = gpuArray(0.01*randn(1,no_classes,'single'));
    net.layers{n+2}.opts = {};
    net.layers{n+2}.dilate = 1;
%     net.layers{21}.type = 'softmaxloss';
end
