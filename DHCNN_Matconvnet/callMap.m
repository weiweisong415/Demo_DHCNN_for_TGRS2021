function [map,index_all] = callMap(Wtrue,Dhamm)
%% Calculate MAP
% Paramter:
% -Wtrue
% -Dhamm
[Ntest,~] = size(Wtrue);
ap = zeros(1,Ntest);
index_all=[];
for j = 1:Ntest
    gnd = Wtrue(j,:);
    tsum = sum(gnd);
    if tsum == 0
        continue;
    end
    ham = Dhamm(j,:);
    [~,index] = sort(ham);
    index_all = [index_all;index];
    gnd = gnd(index);
    count = 1:tsum;
    tindex = find(gnd == 1);
    ap(j) = mean(count./tindex);
end
map = mean(ap,2);