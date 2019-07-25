function output = gaussmf(x,CoreSigma,choice)
%GAUSS Summary of this function goes here
%   Detailed explanation goes here
c=CoreSigma(1);
s=CoreSigma(2);
    switch choice
        %% Theata1
        case 0 %normal
            output=exp(-0.5.*((x-c)./s).*conj((x-c)./s));
        case 1 %dy/dx
            output=exp(-0.5.*((x-c)./s).*conj((x-c)./s)).*-1.*(x-c)./(s.*conj(s));
        case 2 %dy/dc
            output=exp(-0.5.*((x-c)./s).*conj((x-c)./s)).*(x-c)./(s.*conj(s));
        case 3 %dy/da
            output=exp(-0.5.*((x-c)./s).*conj((x-c)./s)).*((x-c).*conj(x-c)./s.^3);
        
        %% Theata2
        case 4 %d2y/dx2
            output=exp(-0.5.*((x-c)./s).^2).*(-1.*(x-c)./s.^2).^2+(-1)./s.^2.*exp(-0.5.*((x-c)./s).^2);
        case 5 %d2y/dc2
            output=exp(-0.5.*((x-c)./s).^2).*(-1.*(x-c)./s.^2).^2+(-1)./s.^2.*exp(-0.5.*((x-c)./s).^2);            
        case 6 %d2y/da2
            output=exp(-0.5.*((x-c)./s).^2).*((x-c).^4./s.^6+(-3.*(x-c).^2)./s.^4);

        case 7 %d2y/dxdc
            output=exp(-0.5.*((x-c)./s).^2).*(1./s.^2+(-(x-c).^2./s.^4));
        case 8 %d2y/dxda
            output=exp(-0.5.*((x-c)./s).^2).*(2.*(x-c)./s.^3+-(x-c).^3./s.^5);            
        case 9 %d2y/dcda
            output=exp(-0.5.*((x-c)./s).^2).*(-2.*(x-c)./s.^3+((x-c).^3./s.^5));
    end

end