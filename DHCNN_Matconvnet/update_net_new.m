function gpu_net = update_net_new (gpu_net, res_back1, res_back2, lr, batch_size, N, eta)
    weight_decay = 5*1e-4 ;
    n_layers = 20 ;
%     batch_size = 128 ;
    for ii = 1:n_layers
            if ~isempty(gpu_net.layers{ii}.weights)
                    weight1 = gpu_net.layers{ii}.weights{1}-...
                        lr*(res_back1(ii).dzdw{1}/(batch_size*N) + weight_decay*gpu_net.layers{ii}.weights{1});
                    weight2 = gpu_net.layers{ii}.weights{1}-...
                        lr*(res_back2(ii).dzdw{1}/(batch_size) + weight_decay*gpu_net.layers{ii}.weights{1});
                    gpu_net.layers{ii}.weights{1} = eta*weight1 + (1-eta)*weight2;
                    bias1 = gpu_net.layers{ii}.weights{2}-...
                        lr*(res_back1(ii).dzdw{2}/(batch_size*N) + weight_decay*gpu_net.layers{ii}.weights{2});
                    bias2 = gpu_net.layers{ii}.weights{2}-...
                        lr*(res_back2(ii).dzdw{2}/(batch_size) + weight_decay*gpu_net.layers{ii}.weights{2});
                    gpu_net.layers{ii}.weights{2} = eta*bias1 + (1-eta)*bias2;
            end
    end
    gpu_net.layers{21}.weights{1} = gpu_net.layers{21}.weights{1}-...
        lr*(res_back2(21).dzdw{1}/(batch_size) + weight_decay*gpu_net.layers{21}.weights{1});
    gpu_net.layers{21}.weights{2} = gpu_net.layers{21}.weights{2}-...
        lr*(res_back2(21).dzdw{2}/(batch_size) + weight_decay*gpu_net.layers{21}.weights{2});
end