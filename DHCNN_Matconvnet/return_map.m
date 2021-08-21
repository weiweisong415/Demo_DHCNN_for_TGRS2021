function map = return_map(B_train, B_test, S)
    [~, orderH] = calcHammingRank (B_train, B_test) ;    
    map = calcMAP(orderH,S');
    disp(map);
    B_train = compactbit(B_train);
    B_test = compactbit(B_test);
    Dhamm = hammingDist(B_test, B_train);
    map = callMap(S', Dhamm);
    disp(map);
end