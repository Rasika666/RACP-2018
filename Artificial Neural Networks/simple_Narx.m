clc;
clear;
%  Here is a problem consisting of sequences of inputs P and targets T
%      that we would like to solve with a network.
 
       P = {[0] [1] [1] [0] [-1] [-1] [0] [1] [1] [0] [-1]};
       T = {[0] [1] [2] [2]  [1]  [0] [1] [2] [1] [0]  [1]};
 
%      Here a two-layer feed-forward network with a two-delay input
%      and two-delay feedback is created.  The hidden layer has 5 neurons.
 
       net = newnarx(P,T,[0 1],[1 2],5);
 
%      Here the network is simulated and its output plotted against
%      the targets.
 
        Y = sim(net,P);
        plot(1:11,[T{:}],1:11,[Y{:}],'o')
 
%      Here the network is trained for 50 epochs.  Again the network's
%       output is plotted.
 
       net = train(net,P,T);
       Yf = sim(net,P);
       plot(1:11,[T{:}],1:11,[Y{:}],'o',1:11,[Yf{:}],'+')
 