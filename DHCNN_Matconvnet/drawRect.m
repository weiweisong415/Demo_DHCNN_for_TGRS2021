function [ output_img ] = drawRect( input_img, pt, wSize,  lineSize, color )
%简介：
% %将图像画上有颜色的框图，如果输入是灰度图，先转换为彩色图像，再画框图
% 图像矩阵
% 行向量方向  是  y
% 列向量方向  是  x
%----------------------------------------------------------------------
%输入：
% input_img：        原始图像，可以为灰度图，可为彩色图
% pt：         左上角坐标   [x1, y1]
% wSize：   框的大小      [wx, wy]
% lineSize： 线的宽度
% color：     线的颜色      [r,  g,  b] 
%----------------------------------------------------------------------
%输出：
% output_img：           画好了的图像


%flag=1: 有缺口的框
%flag=2: 无缺口的框
flag = 2;
%判断输入参数个数
if nargin < 5
    color = [255 255 0];
end
if nargin < 4
    lineSize = 1;
end

[yA, xA, z] = size(input_img);
x1 = pt(1);
y1 = pt(2);
wx = wSize(1);
wy = wSize(2);
%如果是单通道的灰度图，转成3通道的图像
if 1==z
    output_img(:, : ,1) = input_img;
    output_img(:, : ,2) = input_img;
    output_img(:, : ,3) = input_img;
else
    output_img = input_img;
end

%开始画框图
for c = 1 : 3                 %3个通道，r，g，b分别画
    for dl = 1 : lineSize   %线的宽度，线条是向外面扩展的
        d = dl - 1;
        if  1==flag %有缺口的框
            output_img(  y1-d ,            x1:(x1+wx) ,  c  ) =  color(c); %上方线条
            output_img(  y1+wy+d ,     x1:(x1+wx) , c  ) =  color(c); %下方线条
            output_img(  y1:(y1+wy) ,   x1-d ,           c  ) =  color(c); %左方线条
            output_img(  y1:(y1+wy) ,   x1+wx+d ,    c  ) =  color(c); %左方线条
        elseif 2==flag %无缺口的框
            output_img(  y1-d ,            (x1-d):(x1+wx+d) ,  c  ) =  color(c); %上方线条
            output_img(  y1+wy+d ,    (x1-d):(x1+wx+d) ,  c  ) =  color(c); %下方线条
            output_img(  (y1-d):(y1+wy+d) ,   x1-d ,           c  ) =  color(c); %左方线条
            output_img(  (y1-d):(y1+wy+d) ,   x1+wx+d ,    c  ) =  color(c); %左方线条
        end
    end    
end 

end