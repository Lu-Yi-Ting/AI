function [Leader,Convergence_curve,y_hat,TestCost] = SLPSO(num_baseParticle,num_epoch,newC,Test,Train,Target,maxClose,Close1,Days)
Convergence_curve=zeros(1, num_epoch);
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
% Parameter initiliaztion
num_particle = num_baseParticle + floor(nDim/500);
c3 = nDim/num_baseParticle*0.01;
probLearn = (1 - [0:(num_particle-1)]'./num_particle).^log(sqrt(ceil(nDim/num_baseParticle)));

% Paticle initialization
pos = randn(num_particle, nRule*4*nDim);
for i=1:num_particle
    pop(i).theta=zeros(nRule*(nDim+1),nOutput);
end
vel = zeros(num_particle, nRule*4*nDim);
B = zeros(nTrain,nDim+1);
B(:,1)=1;
for k=2:nDim+1
    B(:,k)=Train(k-1).value';
end
Leader.pos=zeros(1,nDim);
Leader.score=inf;
Leader.theta=zeros(nRule*(nDim+1),nOutput);

%% Main loop
for t = 1:num_epoch
    disp(t);
    for i=1:num_particle
        for Dim=1:nDim
            k=1;
            for set=1:length(Train(Dim).core)
                Particle(Dim).fuzzyset(set).CoreSigma=[pos(i,k) pos(i,k+1)];
                Particle(Dim).fuzzyset(set).Lamda=[pos(i,k+2) pos(i,k+3)];
                k=k+4;
            end
        end
        for m=1:nOutput
            Lamda(m).value=SCFS_Normalization(nDim,newC,Train,Particle,m);
        end
        PRMSE=0;
        tmp=1;
        for m=1:nOutput
            A = [];
            for k=1:nRule
                A=[A,B.*repmat(transpose(Lamda(m).value(k,:)),1,size(B,2))];
            end
            pop(i).theta(:,m)=RLSE(A,transpose(y(m).value),pop(i).theta(:,m)); %RLSE
            y_hat{m}=A*pop(i).theta(:,m);    % y=A*theta
            ori_y(m).value=(Close1{tmp}(Days+1:nTrain+Days)+real(y(m).value).*maxClose(tmp))+(Close1{tmp+1}(Days+1:nTrain+Days)+imag(y(m).value).*maxClose(tmp+1))*j;
            ori_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_hat{m}).*maxClose(tmp))+(Close1{tmp+1}(Days+1:nTrain+Days)+imag(y_hat{m}).*maxClose(tmp+1))*j;
            %ori_y(m).value=(Close1{tmp}(Days+1:nTrain+Days)+real(y(m).value).*maxClose(tmp))+(Close1{tmp}(Days+1:nTrain+Days)+imag(y(m).value).*maxClose(tmp))*j;
            %ori_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_hat{m}).*maxClose(tmp))+(Close1{tmp}(Days+1:nTrain+Days)+imag(y_hat{m}).*maxClose(tmp))*j;
            error=transpose(ori_y(m).value)-transpose(ori_y_hat{m});
            PRMSE=PRMSE+RMSE(error,nTrain);  %RMSE
            tmp=tmp+2;
        end
        fitness(i,:)=PRMSE;
    end
    % Population sorting
    for i=1:num_particle   %更新最佳解的cost值
        if fitness(i,:)<=Leader.score  %若某位置的cost小於等於目前最佳的cost即取代之
            Leader.pos=pos(i, :);
            Leader.theta=pop(i).theta;
            Leader.score=fitness(i,:);
        end
    end
    [fitness, rank] = sort(fitness, 'descend');
    pos = pos(rank, :);
    vel = vel(rank, :);
    
    
    % Demonstrator
    demoIndexMask = [1:num_particle]';
    demoIndex = demoIndexMask + ceil(rand(num_particle, nRule*4*nDim).*(num_particle - demoIndexMask));
    demonstrator = pos;
    for n = 1:(nRule*4*nDim)
        demonstrator(:,n) = pos(demoIndex(:,n),n);
    end
    
    % Collective behavior
    center = mean(pos);
    
    % Random matrix
    randco1 = rand(num_particle, nRule*4*nDim);
    randco2 = rand(num_particle, nRule*4*nDim);
    randco3 = rand(num_particle, nRule*4*nDim);
    
    % Social learning
    lpmask = rand(num_particle,1) < probLearn;
    lpmask(end) = 0;
    
    v1 =  randco1.*vel + randco2.*(demonstrator - pos) + c3*randco3.*(center - pos);
    p1 =  pos + v1;
    
    vel = lpmask.*v1 + (~lpmask).*vel;
    pos = lpmask.*p1 + (~lpmask).*pos;
    
    Convergence_curve(t) = Leader.score;
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


