function [Hfj,Hfifj] = entropy_condition(fi,fj,index) 
    fx=fi; % x is the condition
    fy=fj;
    for i=1:length(index)
        fyx(i,1)=fy(index(i));
    end
    [pdx,deltaX]=Pd(fx);
    [pdy,deltaY]=Pd(fy);
    [pdyx,deltaYX]=Pd(fyx);
    a=[pdy,pdyx,pdx];
    phi=max(max(a),1);
    Hfifj=sum(pdx.*sum(pdyx.*log10(phi./pdyx))).*deltaYX.*deltaX;
    Hfj=sum(pdy.*log10(phi./pdy)).*deltaY;
end
