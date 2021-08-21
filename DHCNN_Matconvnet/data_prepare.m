function [data, label] = data_prepare(img_dir, dataset_name, img_type)
subfolders = dir(img_dir);
data = [];
label = [];
kk=1;
for ii = 1:length(subfolders)
    subname = subfolders(ii).name;
    if ~strcmp(subname, '.') && ~strcmp(subname, '..')
        frames = dir(fullfile(img_dir, subname,img_type));
        c_num = length(frames); 
        for jj = 1:c_num
            imgpath = fullfile(img_dir, subname, frames(jj).name);
            I = imread(imgpath);
            if size(I,1)==256 && size(I,2)==256   %% for AID: 600; UCM21: 256
                data=cat(4, data, I);
            else
                 I = imresize(I, [256,256],'bicubic') ;
                 data=cat(4, data, I);
            end
%             [h,w,b]=size(I);
%             data{kk}{1}=h;
%             data{kk}{2}=w;
%             data{kk}{3}=b;
%             data{kk}=I;
%             kk=kk+1;
%             I = single(I) ; % note: 255 range
%            I = imresize(I, net.meta.normalization.imageSize(1:2),'bicubic') ;
%            I = I -net.meta.normalization.averageImage ;
%             data=cat(4, data, I);

        end
        lab_tem = ones(1, c_num)*(ii-2);
        label = cat(2, label, lab_tem);
    end
end
% data = double(data);
save(strcat(dataset_name, '.mat'), 'data','-v7.3', 'label');
end
