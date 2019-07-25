function lamda=SCFS_Normalization(nDim,newC,Train,Input,nOutput)
for rule=1:size(newC,1)
    membership=1;
    for n=1:nDim
        r=gaussmf(Train(n).value,Input(n).fuzzyset(newC(rule,n)).CoreSigma,0);         
        theata1Ofh=gaussmf(Train(n).value,Input(n).fuzzyset(newC(rule,n)).CoreSigma,2)*Input(n).fuzzyset(newC(rule,n)).Lamda(1); %dr/dx
        theata2Ofh=gaussmf(Train(n).value,Input(n).fuzzyset(newC(rule,n)).CoreSigma,7)*Input(n).fuzzyset(newC(rule,n)).Lamda(2); %d2r/dx^2
            switch nOutput
                case 1
                    temp=(r.*cos(theata2Ofh).*cos(theata1Ofh))+j*(r.*cos(theata2Ofh).*sin(theata1Ofh));
                case 2
                    temp=(r.*cos(theata2Ofh).*cos(theata1Ofh))+j*(r.*sin(theata2Ofh));
                case 3
                    temp=(r.*cos(theata2Ofh).*sin(theata1Ofh))+j*(r.*sin(theata2Ofh));
                case 4
                    temp=r.*exp(j*theata1Ofh);
                case 5
                    temp=r.*exp(j*theata2Ofh);
                case 6
                    temp=r.*exp(j*(theata1Ofh+theata2Ofh));
            end
        membership=membership.*temp;
    end
    NewBeta(rule,:)=membership;    
end
normalbeta=NewBeta./repmat(sum(NewBeta),size(NewBeta,1),1);
normalbeta(isnan(normalbeta))=0;
lamda=normalbeta;