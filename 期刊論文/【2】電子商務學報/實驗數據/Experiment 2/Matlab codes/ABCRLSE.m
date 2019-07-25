function [Leader,Convergence_curve,y_hat,TestCost]=ABCRLSE(SN,MCN,Onlooker,limit,newC,Test,Train,Target,maxClose,Close1,Days)
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
Convergence_curve=zeros(1,MCN);
%% Initialization
Abandon=zeros(SN,1);     %�����ӭ������Q�^�O������

for i=1:SN
    pop(i).Position=randn(1, nRule*4*nDim);   %�b�j���Ŷ����H���w��
    pop(i).theta=zeros(nRule*(nDim+1),nOutput);
end
newbee.Cost=inf;
newbee.theta=zeros(nRule*(nDim+1),nOutput);
B = zeros(nTrain,nDim+1);
B(:,1)=1;
for k=2:nDim+1
    B(:,k)=Train(k-1).value';
end

%% Main loop 
for n=1:MCN 
    disp(n);
    for i=1:SN
        for Dim=1:nDim
            k=1;
            for set=1:length(Train(Dim).core)
                Bee(Dim).fuzzyset(set).CoreSigma=[pop(i).Position(k) pop(i).Position(k+1)];
                Bee(Dim).fuzzyset(set).Lamda=[pop(i).Position(k+2) pop(i).Position(k+3)];
                k=k+4;
            end
        end
        for m=1:nOutput
            Lamda(m).value=SCFS_Normalization(nDim,newC,Train,Bee,m);
        end
        BeeRMSE=0;
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
            BeeRMSE=BeeRMSE+RMSE(error,nTrain);  %RMSE
            tmp=tmp+2;
        end
        pop(i).Cost=BeeRMSE;
        
        K=[1:i-1 i+1:SN];
        k=randi([1 length(K)]);       %�H����ܤ@���Di���u��k 
        newbee.Position=pop(i).Position+unifrnd(-1,+1).*(pop(i).Position-pop(k).Position);  %�ѦҤu��k����m����  
        for Dim=1:nDim
            k=1;
            for set=1:length(Train(Dim).core)
                NewBee(Dim).fuzzyset(set).CoreSigma=[newbee.Position(k) newbee.Position(k+1)];
                NewBee(Dim).fuzzyset(set).Lamda=[newbee.Position(k+2) newbee.Position(k+3)];
                k=k+4;
            end
        end
        for m=1:nOutput
            Lamda(m).value=SCFS_Normalization(nDim,newC,Train,NewBee,m);
        end
        NewBeeRMSE=0;
        tmp=1;
        for m=1:nOutput
            A = [];
            for k=1:nRule
                A=[A,B.*repmat(transpose(Lamda(m).value(k,:)),1,size(B,2))];
            end
            newbee.theta(:,m)=RLSE(A,transpose(y(m).value),newbee.theta(:,m)); %RLSE
            y_newbee{m}=A*newbee.theta(:,m);    % y=A*theta
            ori_new_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_newbee{m}).*maxClose(tmp))+(Close1{tmp+1}(Days+1:nTrain+Days)+imag(y_newbee{m}).*maxClose(tmp+1))*j;
           %ori_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_hat{m}).*maxClose(tmp))+(Close1{tmp}(Days+1:nTrain+Days)+imag(y_hat{m}).*maxClose(tmp))*j;
            error=transpose(ori_y(m).value)-transpose(ori_new_y_hat{m});
            NewBeeRMSE= NewBeeRMSE+RMSE(error,nTrain);  %RMSE %�������ʫ��m��cost��
            tmp=tmp+2;
        end
        newbee.Cost=NewBeeRMSE;
        if newbee.Cost<=pop(i).Cost   %�Y�s��m��cost�Ȥp�󵥩�쥻����m�h����
            pop(i)=newbee;
        else
            Abandon(i)=Abandon(i)+1;  %�ϫh�N���ʵL�k���Ccost���ȡA���ಣ�ͷs�����q�A�G�^�O�ƥ[�@
        end        
    end   
    %% �p��A���ȩM���v(���L�k)
    Fit=zeros(SN,1);
    MeanCost = mean([pop.Cost]);
    for i=1:SN
        Fit(i) = exp(-pop(i).Cost/MeanCost); % Convert Cost to Fitness
    end
    P=Fit/sum(Fit);   %�[�����ܸ��I�����v�O��P
    for m=1:Onlooker  %���X�[���
        i=RouletteWheelSelection(P);%�ϥν��L�k�M�w�[������a�I
        if isnan(i)==1
            i=randi([1 SN]);
        end
        if isempty(i)==1
            i=randi([1 SN]);
        end
        disp('i=');
        disp(i);
        K=[1:i-1 i+1:SN];
        %���ƤW�z�����u�������k
        k=K(randi([1 numel(K)]));
        newbee.Position=pop(i).Position+unifrnd(-1,+1).*(pop(i).Position-pop(k).Position);
        for Dim=1:nDim
            k=1;
            for set=1:length(Train(Dim).core)
                NewBee(Dim).fuzzyset(set).CoreSigma=[newbee.Position(k) newbee.Position(k+1)];
                NewBee(Dim).fuzzyset(set).Lamda=[newbee.Position(k+2) newbee.Position(k+3)];
                k=k+4;
            end
        end
        for m=1:nOutput
            Lamda(m).value=SCFS_Normalization(nDim,newC,Train,NewBee,m);
        end
        NewBeeRMSE=0;
        tmp=1;
        for m=1:nOutput
            A = [];
            for k=1:nRule
                A=[A,B.*repmat(transpose(Lamda(m).value(k,:)),1,size(B,2))];
            end
            newbee.theta(:,m)=RLSE(A,transpose(y(m).value),newbee.theta(:,m)); %RLSE
            y_newbee{m}=A*newbee.theta(:,m);    % y=A*theta
            ori_new_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_newbee{m}).*maxClose(tmp))+(Close1{tmp+1}(Days+1:nTrain+Days)+imag(y_newbee{m}).*maxClose(tmp+1))*j;
           %ori_y_hat{m}=(Close1{tmp}(Days+1:nTrain+Days)+real(y_hat{m}).*maxClose(tmp))+(Close1{tmp}(Days+1:nTrain+Days)+imag(y_hat{m}).*maxClose(tmp))*j;
            error=transpose(ori_y(m).value)-transpose(ori_new_y_hat{m});
            NewBeeRMSE= NewBeeRMSE+RMSE(error,nTrain);  %RMSE %�������ʫ��m��cost��
            tmp=tmp+2;
        end
        newbee.Cost=NewBeeRMSE;
        if newbee.Cost<=pop(i).Cost   %�Y�s��m��cost�Ȥp�󵥩�쥻����m�h����
            pop(i)=newbee;
        else
            Abandon(i)=Abandon(i)+1;  %�ϫh�N���ʵL�k���Ccost���ȡA���ಣ�ͷs�����q�A�G�^�O�ƥ[�@
        end  
    end

    for i=1:SN                         
        if Abandon(i)>=limit   %�Y�^�O�Ƥj�󷥭��ȧY���X���d���A�b���줤�H���w��
            pop(i).Position=randn(1, nRule*4*nDim);   %�b�j���Ŷ����H���w��
            pop(i).theta=zeros(nRule*(nDim+1),nOutput);
            Abandon(i)=0;
        end
    end

    for i=1:SN   %��s�̨θѪ�cost��
        if pop(i).Cost<=Leader.score  %�Y�Y��m��cost�p�󵥩�ثe�̨Ϊ�cost�Y���N��
            Leader.pos=pop(i).Position;
            Leader.theta=pop(i).theta;
            Leader.score=pop(i).Cost;
        end
    end
    Convergence_curve(n)=Leader.score;
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
    fitness(m,:)=RMSE(error,size(Testy_hat{m},1));  %RMSE
    y_hat{m}=[y_hat{m};Testy_hat{m}];
    tmp=tmp+2;
end
TestCost=fitness;
end