function [Leader,Convergence_curve,y_hat,TestCost] = CACO(opts,newC,Test,Train,Target,maxClose,Close1,Days)
%% Transfer option varible
iteration = opts.iteration;
nAnt = opts.nAnt;
new_nAnt = opts.new_nAnt;
all_nAnt = nAnt + new_nAnt;
nDim = opts.nDim;
learning_rate = opts.learning_rate;
nRule=size(newC,1);
nTarget=size(Target,2);
nOutput=ceil(nTarget/2);
nTrain=size(Train(1).value,2);
i=1;
for t=1:nOutput
    if nOutput>=2        
        realPart=Target{i};
        imagPart=Target{i+1};
        Testy(t).value=realPart+imagPart*j;
        y(t).value=Testy(t).value(1:nTrain);
        i=i+2;
    else
        realPart=Target{1};
        imagPart=Target{1};
        Testy(t).value=realPart+imagPart*j;
        y(t).value=Testy(t).value(1:nTrain);
    end
end
Leader.pos=zeros(1,nDim);
Leader.score=inf;
Leader.theta=zeros(nRule*(nDim+1),nOutput);

RMSE=@(error,n) ((error*error')/n).^0.5;
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
weight = gauF(1:nAnt, 1, learning_rate * nAnt) ./ (learning_rate * nAnt * sqrt(2*pi));
opts.prob = weight ./ sum(weight);

%% Initial
for i=1:nAnt
    ant(i).Position=randn(1, nRule*4*nDim);   %在搜索空間中隨機定位
    ant(i).theta=zeros(nRule*(nDim+1),nOutput);
end
B = zeros(nTrain,nDim+1);
B(:,1)=1;
for k=2:nDim+1
    B(:,k)=Train(k-1).value';
end
for i=1:nAnt
    ant(i).Position=randn(1, nRule*4*nDim);   %在搜索空間中隨機定位
    ant(i).theta=zeros(nRule*(nDim+1),nOutput);
end
for i=(1:new_nAnt) + nAnt
    ant(i).theta=zeros(nRule*(nDim+1),nOutput);
end
%% Training
Convergence_curve = zeros(1, iteration);
for n = 1:iteration
    disp(n);
    %% New ant and calculate cost
    ant = antrenew(ant, opts);
    for i=1:(new_nAnt + nAnt)
        for Dim=1:nDim
            k=1;
            for set=1:length(Train(Dim).core)
                Ant(Dim).fuzzyset(set).CoreSigma=[ant(i).Position(k) ant(i).Position(k+1)];
                Ant(Dim).fuzzyset(set).Lamda=[ant(i).Position(k+2) ant(i).Position(k+3)];
                k=k+4;
            end
        end
        for m=1:nOutput
            Lamda(m).value=SCFS_Normalization(nDim,newC,Train,Ant,m);
        end
        AntRMSE=0;
        tmp=1;
        for m=1:nOutput
            A = [];
            for k=1:nRule
                A=[A,B.*repmat(transpose(Lamda(m).value(k,:)),1,size(B,2))];
            end
            ant(i).theta(:,m)=RLSE(A,transpose(y(m).value),ant(i).theta(:,m)); %RLSE
            y_hat{m}=A*ant(i).theta(:,m);    % y=A*theta
            ori_y(m).value=(Close1{tmp}(Days+1:nTrain+Days)+real(y(m).value).*maxClose(tmp))+(Close1{tmp+1}(Days+1:nTrain+Days)+imag(y(m).value).*maxClose(tmp+1))*j;
            ori_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_hat{m}).*maxClose(tmp))+(Close1{tmp+1}(Days+1:nTrain+Days)+imag(y_hat{m}).*maxClose(tmp+1))*j;
            %ori_y(m).value=(Close1{tmp}(Days+1:nTrain+Days)+real(y(m).value).*maxClose(tmp))+(Close1{tmp}(Days+1:nTrain+Days)+imag(y(m).value).*maxClose(tmp))*j;
            %ori_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_hat{m}).*maxClose(tmp))+(Close1{tmp}(Days+1:nTrain+Days)+imag(y_hat{m}).*maxClose(tmp))*j;
            error=transpose(ori_y(m).value)-transpose(ori_y_hat{m});
            AntRMSE=AntRMSE+RMSE(error,nTrain);  %RMSE
            tmp=tmp+2;
        end
        ant(i).Cost=AntRMSE;        
    end
    for i=1:nAnt   %更新最佳解的cost值
        if ant(i).Cost<=Leader.score  %若某位置的cost小於等於目前最佳的cost即取代之
            Leader.pos=ant(i).Position;
            Leader.theta=ant(i).theta;
            Leader.score=ant(i).Cost;
        end
    end
    
    Convergence_curve(n) = Leader.score;
end
for Dim=1:nDim
    k=1;
    for number=1:length(Train(Dim).core)
        TestInput(Dim).fuzzyset(number).CoreSigma=[Leader.pos(k) Leader.pos(k+1)];
        TestInput(Dim).fuzzyset(number).Lamda=[Leader.pos(k+2) Leader.pos(k+3)];
        k=k+4;
    end
end
for m=1:nOutput
    Lamda(m).value=SCFS_Normalization(nDim,newC,Test,TestInput,m);
end
B = zeros(length(Test(1).value),nDim);
B(:,1)=1;
for n=2:nDim+1
    B(:,n)=Test(n-1).value';
end
fitness=0;
tmp=1;
for m=1:nOutput
    A = [];
    for r=1:nRule
        A=[A,B.*repmat(transpose(Lamda(m).value(r,:)),1,size(B,2))];
    end
    Testy_hat{m}=A*Leader.theta(:,m);    % y=A*theta
   ori_Testy(m).value=(Close1{tmp}(nTrain+Days+1:end)+real(Testy(m).value(nTrain+1:end)).*maxClose(tmp))+(Close1{tmp+1}(nTrain+Days+1:end)+imag(Testy(m).value(nTrain+1:end)).*maxClose(tmp+1))*j;
   ori_Testy_hat{m}=(Close1{tmp}(nTrain+Days+1:end)+real(Testy_hat{m}).*maxClose(tmp))+(Close1{tmp+1}(nTrain+Days+1:end)+imag(Testy_hat{m}).*maxClose(tmp+1))*j;

%   ori_Testy(m).value=(Close1{tmp}(nTrain+Days+1:end)+real(Testy(m).value(nTrain+1:end)).*maxClose(tmp))+(Close1{tmp}(nTrain+Days+1:end)+imag(Testy(m).value(nTrain+1:end)).*maxClose(tmp))*j;
%   ori_Testy_hat{m}=(Close1{tmp}(nTrain+Days+1:end)+real(Testy_hat{m}).*maxClose(tmp))+(Close1{tmp}(nTrain+Days+1:end)+imag(Testy_hat{m}).*maxClose(tmp))*j;
    error=transpose(ori_Testy(m).value)-transpose(ori_Testy_hat{m});
    fitness=fitness+RMSE(error,size(Testy_hat{m},1));  %RMSE
    y_hat{m}=[y_hat{m};Testy_hat{m}];
    tmp=tmp+2;
end
TestCost=fitness;
end