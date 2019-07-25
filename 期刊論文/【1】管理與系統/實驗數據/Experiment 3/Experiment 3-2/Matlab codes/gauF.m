function MF = gauF(x, center, sigma) 
MF = exp(-0.5.*((x-center)./sigma).*conj((x-center)./sigma));
