function [map,OA, recall,precision, rec, pre] = test_retrieval(net, dataset_L, test_L,data_set, test_data, no_classes, pos )
    S = compute_S(dataset_L,test_L) ;
    [B_dataset, B_test, Pre_L_dataset, Pre_L_test] = compute_B (data_set,test_data,net) ;
    [OA,Kappa,AA,CA, cfm] = calcError(test_L-1,Pre_L_test-1,[1:no_classes]);
    
    B_dataset = compactbit(B_dataset);
    B_test = compactbit(B_test);
    
    Dhamm = hammingDist(B_test, B_dataset);
    [map,~] = callMap(S', Dhamm);
    [recall, precision, ~] = recall_precision(S', Dhamm);
    [rec, pre]= recall_precision5(S', Dhamm, pos); % recall VS. the number of retrieved sample
end
