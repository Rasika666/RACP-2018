%Sample Training Session from MatLab

clc;
clear;

load choles_all;
[pn,ps1] = mapstd(p);
[ptrans,ps2] = processpca(pn,0.001);
[tn,ts] = mapstd(t);
[R,Q] = size(ptrans);
iitst = 2:4:Q;
iival = 4:4:Q;
iitr = [1:4:Q 3:4:Q];
val.P = ptrans(:,iival); val.T = tn(:,iival);
test.P = ptrans(:,iitst); test.T = tn(:,iitst);
teP=ptrans(:,iitst); teT=tn(:,iitst);
ptr = ptrans(:,iitr); ttr = tn(:,iitr);
net = newff(minmax(ptr),[5 3],{'tansig' 'purelin'},'trainlm');
[net,tr]=train(net,ptr,ttr,[],[],val,test);
plot(tr.epoch,tr.perf,tr.epoch,tr.vperf,tr.epoch,tr.tperf)
legend('Training','Validation','Test',-1);
ylabel('Squared Error'); xlabel('Epoch');
an = sim(net,ptrans);
a = mapstd('reverse',an,ts);
for i=1:3
  figure(i)
  [m(i),b(i),r(i)] = postreg(a(i,:),t(i,:));
end

A=sim(net,teP);
N=1:66;
for j=1:3
    dif(j,:)=teT(j,:)-A(j,:);
end

