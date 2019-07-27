function[Leader,Convergence_curve,y_hat,TestCost]=WOARLSE(SearchAgents,newC,Iteration,Test,Train,Target,maxClose,Close1,Days)
RMSE=@(error,n) ((error*error')/n).^0.5;
nDim=size(Test,2);
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
for i=1:SearchAgents
    Agent(i).Position=randn(1, nRule*4*nDim);
    Agent(i).theta=zeros(nRule*(nDim+1),nOutput);
end
B = zeros(nTrain,nDim+1);
B(:,1)=1;
for k=2:nDim+1
    B(:,k)=Train(k-1).value';
end
DimPosition=[];
for i=1:SearchAgents
    DimPosition=[DimPosition;Agent(i).Position];
end
DimPosition=reshape(DimPosition,numel(DimPosition),1);
%%  Main loop
for t=1:Iteration
    disp(t);
    for i=1:SearchAgents
        for Dim=1:nDim
            k=1;
            for set=1:length(Train(Dim).core)
                Input(Dim).fuzzyset(set).CoreSigma=[Agent(i).Position(k) Agent(i).Position(k+1)];
                Input(Dim).fuzzyset(set).Lamda=[Agent(i).Position(k+2) Agent(i).Position(k+3)];
                k=k+4;
            end
        end
        for m=1:nOutput
            Lamda(m).value=SCFS_Normalization(nDim,newC,Train,Input,m);
        end
        fitness=0;
        tmp=1;
        for m=1:nOutput
            A = [];
            for k=1:nRule
                A=[A,B.*repmat(transpose(Lamda(m).value(k,:)),1,size(B,2))];
            end
            Agent(i).theta(:,m)=RLSE(A,transpose(y(m).value),Agent(i).theta(:,m)); %RLSE
            y_hat{m}=A*Agent(i).theta(:,m);    % y=A*theta
             ori_y(m).value=(Close1{tmp}(Days+1:nTrain+Days)+real(y(m).value).*maxClose(tmp))+(Close1{tmp+1}(Days+1:nTrain+Days)+imag(y(m).value).*maxClose(tmp+1))*j;
             ori_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_hat{m}).*maxClose(tmp))+(Close1{tmp+1}(Days+1:nTrain+Days)+imag(y_hat{m}).*maxClose(tmp+1))*j;
%           ori_y(m).value=(Close1{tmp}(Days+1:nTrain+Days)+real(y(m).value).*maxClose(tmp))+(Close1{tmp}(Days+1:nTrain+Days)+imag(y(m).value).*maxClose(tmp))*j;
%           ori_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_hat{m}).*maxClose(tmp))+(Close1{tmp}(Days+1:nTrain+Days)+imag(y_hat{m}).*maxClose(tmp))*j;
             error=transpose(ori_y(m).value)-transpose(ori_y_hat{m});
            fitness=fitness+RMSE(error,nTrain);  %RMSE
            tmp=tmp+2;
        end
        if fitness<Leader.score % Change this to > for maximization problem
            Leader.score=fitness; % Update alpha
            Leader.pos=Agent(i).Position;
            Leader.theta=Agent(i).theta;
        end
    end
    
    a=2-t*((2)/Iteration);
    a2=-1+t*((-1)/Iteration);
    
    for i=1:SearchAgents
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
        p = rand();        % p in Eq. (2.6)        
        
        if p<0.5
            if abs(A)>=1
                rand_leader_index = floor(SearchAgents*rand()+1);
                X_rand = Agent(rand_leader_index).Position;
                D_X_rand=abs(C*X_rand(m)-Agent(i).Position); % Eq. (2.7)
                Agent(i).Position=X_rand-A*D_X_rand;      % Eq. (2.8)
            elseif abs(A)<1
                D_Leader=abs(C*Leader.pos-Agent(i).Position); % Eq. (2.1)
                Agent(i).Position=Leader.pos + A.*D_Leader;
            end
        else
            distance2Leader=abs(Leader.pos-Agent(i).Position);% Eq. (2.5)
            Agent(i).Position=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader.pos;
         end
    end
    Convergence_curve(t)=Leader.score;
end
%% Testing
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
    %% Testing RMSE for selection
     ori_Testy(m).value=(Close1{tmp}(nTrain+Days+1:end)+real(Testy(m).value(nTrain+1:end)).*maxClose(tmp))+(Close1{tmp+1}(nTrain+Days+1:end)+imag(Testy(m).value(nTrain+1:end)).*maxClose(tmp+1))*j;
     ori_Testy_hat{m}=(Close1{tmp}(nTrain+Days+1:end)+real(Testy_hat{m}).*maxClose(tmp))+(Close1{tmp+1}(nTrain+Days+1:end)+imag(Testy_hat{m}).*maxClose(tmp+1))*j;
%   ori_Testy(m).value=(Close1{tmp}(nTrain+Days+1:end)+real(Testy(m).value(nTrain+1:end)).*maxClose(tmp))+(Close1{tmp}(nTrain+Days+1:end)+imag(Testy(m).value(nTrain+1:end)).*maxClose(tmp))*j;
%   ori_Testy_hat{m}=(Close1{tmp}(nTrain+Days+1:end)+real(Testy_hat{m}).*maxClose(tmp))+(Close1{tmp}(nTrain+Days+1:end)+imag(Testy_hat{m}).*maxClose(tmp))*j;
    error=transpose(ori_Testy(m).value)-transpose(ori_Testy_hat{m});
    fitness=fitness+RMSE(error,size(Testy_hat{m},1));  %RMSE
    y_hat{m}=[y_hat{m};Testy_hat{m}];
    %% Testing RMSE for each stock
    ori_stock{tmp}=real(ori_Testy(m).value);
    ori_stock{tmp+1}=imag(ori_Testy(m).value);
    ori_stock_yhat{tmp}=real(ori_Testy_hat{m});
    ori_stock_yhat{tmp+1}=imag(ori_Testy_hat{m});
    tmp=tmp+2;
end
TestCost=fitness;
