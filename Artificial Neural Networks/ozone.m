% ********************** DATA FOR ASSIGNMENT1 *************************************
% This program reads 'ozone.csv' data file, devide the full
% data set into 3 sets.
% 
% Each data set is seved as separate file.
%--------------------------------------------------------------------------


clear; %To clear the workspace
clc; %To clear the command window

%Read csv data file into a matrix
M = dlmread('E:\DataMining\ozone.csv', ',', 'B2..G331');
[r,c]=size(M);

for i=1:r
    T(i,1)=M(i,1);
    for k=1:5
        P(i,k)=M(i,k+1);
    end
    p=P';
    t=T';
end
 
[trainV,valV,testV] = dividevec(p,t,0.20,0.10);

net = newff(minmax(p),[2 1],{'tansig' 'purelin'},'trainlm');
net.trainParam.epochs = 300;
net.trainParam.lr = 0.01;
net.trainParam.mc = 0.03;
[net,tr,Y,E] = train(net,trainV.P,trainV.T,[],[],valV,testV);

for i=1:length(testV)
    testT=testV.T;
    testP=testV.P;
end
    

[Pred]=sim(net,testP);
for n=1:length(testP);
    TError(n)=testT(n)-Pred(n);
end
   plot(TError);
   hgsave('E:\DataMining\eplot');
