function IIM = InfluenceMatrix(DataMatrix)
iim=zeros(size(DataMatrix,2));
Entropy=@(phi,pd,delta) sum(pd.*log10(phi./pd)).*delta;
Entropy_condition=@(phi,pdx,pdyx,deltax,deltayx) sum(pdx.*sum(pdyx.*log10(phi./pdyx))).*deltax.*deltayx;

for i=1:size(DataMatrix,2)
    fi=DataMatrix(:,i);
    negative=DataMatrix(DataMatrix(:,i)<0,i);
    positive=DataMatrix(DataMatrix(:,i)>=0,i);
    
    [pdn,dn]=Pd(negative);
    [pdp,dp]=Pd(positive);
    pdi=fitdist(fi,'kernel');
    rat1=cdf(pdi,0);
    rat2=1-rat1;

    for j=1:size(DataMatrix,2)
        fj=DataMatrix(:,j);
        nj=DataMatrix(DataMatrix(:,i)<0,j);
        pj=DataMatrix(DataMatrix(:,i)>=0,j);
        
        [pdj,dj]=Pd(fj);
        [pdnj,dnj]=Pd(nj);
        [pdpj,dpj]=Pd(pj);
        
        phi=max([pdj,pdnj,pdpj,1]);
        
        Hfj=Entropy(phi,pdj,dj);
        Hnj=Entropy_condition(phi,pdn,pdnj,dn,dnj);
        Hpj=Entropy_condition(phi,pdp,pdpj,dp,dpj);
        
        if i==j
            iim(i,j)=0;
        else
            iim(i,j)=(Hfj-Hnj)*rat1+(Hfj-Hpj)*rat2;
        end 
    end
end

IIM=iim;