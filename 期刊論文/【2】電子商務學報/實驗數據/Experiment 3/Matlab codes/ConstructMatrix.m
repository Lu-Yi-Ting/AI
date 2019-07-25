function newC=ConstructMatrix(ConIndex,train_h)
InputNum=size(train_h,2);
Rule_Accu=zeros(1,length(ConIndex));
for rule=1:size(ConIndex,1)
    membership=1;
    for n=1:InputNum
        temp=gauF(train_h(n).value,train_h(n).core(ConIndex(rule,n)),train_h(n).sigma);
        membership=membership.*temp;
    end
    Rule_Accu(rule)=Rule_Accu(rule)+sum(membership); 
end
new = ConIndex*0;
avg=mean(Rule_Accu);
n=1;
for i=1:size(ConIndex,1)
    if Rule_Accu(i)>avg
        new(n,:)=ConIndex(i,:);
        newaccu(n)=Rule_Accu(i);
        n=n+1;
    end
end

new(n:end,:)=[];

nrule=size(new,1);
maxrule=3;
minrule=2;
newCrule=min(max(minrule,nrule),maxrule);
if newCrule==minrule
    nrule=size(ConIndex,1);
    [Rule_Accu,index]=sort(Rule_Accu,'descend');
    index=index(1:min(max(minrule,nrule),maxrule));
    newC=ConIndex(index);
elseif newCrule==nrule
    newC=new;
elseif newCrule==maxrule
    [newaccu,index]=sort(newaccu,'descend');
    index=index(1:maxrule);
    newC=new(index,:);
end


