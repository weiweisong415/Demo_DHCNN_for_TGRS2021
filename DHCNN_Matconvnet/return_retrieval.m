function  OA = return_retrieval(net, train_L, query_L,train_data, query_data,num_neib, no_classes )

    S = compute_S(train_L,query_L) ;
    [B_dataset, B_query, Pre_L_dataset, Pre_L_query] = compute_B (train_data,query_data,net) ;
    [OA,Kappa,AA,CA, cfm] = calcError(query_L-1,Pre_L_query-1,[1:no_classes]);
    B_dataset = compactbit(B_dataset);
    B_query = compactbit(B_query);
    
    Dhamm = hammingDist(B_query, B_dataset);
    [~,index] = callMap(S', Dhamm);
    top_index = index(:,1:num_neib);
    Dir = 'maps/UCM21/DHCNN_backup/query';
    for kk=1:no_classes
        dir = strcat(Dir,num2str(kk),'/*.jpg');
        delete(dir);
    end
    for ii=1:size(query_data,4)
        I= query_data(:,:,:,ii);
        true_label = query_L(ii);
%         imshow(I);
        im_dir = strcat(Dir,num2str(ii),'/Query_img_TL_',num2str(query_L(ii)),'_PL_',num2str(Pre_L_query(ii)),'.jpg');
        imwrite(I,im_dir,'jpg');
        
        for jj=1:num_neib
            neib_id = top_index(ii,jj);
            I_neib = train_data(:,:,:,neib_id);
            turn_label = Pre_L_dataset(neib_id);
            if true_label == train_L(neib_id)
                I_neib = drawRect(I_neib, [6,6], [size(I_neib,1)-11, size(I_neib,2)-11], 6, [0,255,0]);
            else
                I_neib = drawRect(I_neib, [6,6], [size(I_neib,1)-11, size(I_neib,2)-11], 6, [255,0,0]);
            end
%             imshow(I_neib);
            neib_dir = strcat(Dir,num2str(ii),'/Returned_img',num2str(jj),'_TL_',num2str(train_L(neib_id)),'_PL_',num2str(Pre_L_dataset(neib_id)),'.jpg' );
            imwrite(I_neib,neib_dir,'jpg');
        end
    end
end
