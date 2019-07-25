function FP=FeatureSelection(IIM,nTarget,nFeature,Selection)
target=size(IIM,2);
for i=1:target
    sp(i).value=[];
    sp(i).gain=[];
end
for i=1:target
    for fi=1:length(IIM{i})-1
        re=0;
        if ~isempty(sp(i).value)
            for fj=1:length(sp(i).value)
                re=re+IIM{i}(fi,sp(i).value(fj))+IIM{i}(sp(i).value(fj),fi);
            end
            re=re/fj/2;
        end
        I=IIM{i}(fi,end);
        gain=I-re;
        if gain>0
            sp(i).value=[sp(i).value,fi];
            sp(i).gain=[sp(i).gain,gain];
        end
    end
end
all=[];
for i=1:target
    all=[all,sp(i).value];
end
Omega=unique(all);
nOL=hist(all,Omega);
omega=nOL./target;
GainSum(1:length(Omega))=0;
for i=1:length(Omega)
    for t=1:target
        for j=1:length(sp(t).value)
            if Omega(i)==sp(t).value(j)
                GainSum(i)= GainSum(i)+sp(t).gain(j);
            end
        end
    end
end
% MeanGainSum=mean(GainSum);
% MeanW=mean(omega);
% Rhoth=MeanW*MeanGainSum;
% ntmp=0;
for i=1:length(Omega)
    Rho(i)=omega(i)*GainSum(i);
end
ntmp=length(Omega);
% nL=TargetNum*10;nU=TargetNum*20;
% nL=4;nU=10;
[Rho,index]=sort(Rho,Selection);
nFP=nFeature;
%nFP=max(min(nU,ntmp),nL);
% nInput=2;
%index=index(randperm(nFP,nInput));
index=index(1:nFP);
FP=Omega(index);