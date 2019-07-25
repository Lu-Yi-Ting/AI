function [pdx,delta,x] = Pd(f) 
    feature=f;
    pd=fitdist(feature,'Kernel');
    x=linspace(pd.mean-5*pd.std,pd.mean+5*pd.std,500);
    delta=x(2)-x(1);
    pdx=pdf(pd,x); 
end
