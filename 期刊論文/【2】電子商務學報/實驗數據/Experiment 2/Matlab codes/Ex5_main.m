clear;clc;close all;
RMSE=@(error,n) ((error*error')/n).^0.5;
execute=10;
Iteration=1000;
filename={'^IXIC(2018).csv','NASDAQ'
    '^N225(2018).csv','Nikkei'
     '000001.SS(2018).csv','SSEC'
     '^HSI(2018).csv','HSI'};
algorithm={'WOARLSE','WOA','ABCRLSE','ABC','CACORLSE','CACO','SLPSORLSE','SLPSO'};
nFeature=10;
Selection='descend';
for z=1:size(algorithm,2)
    Closebefore=[];
    Closeafter=[];
    Gainse=[];
    Close1=[];
    Close2=[];
    DataMatrix=[];
    nTarget=size(filename,1);
    for i=1:nTarget
        Stockdata{i}=csvread(filename{i,1},1,0);
    end
    date=Stockdata{i}(:,1);
    for i=1:nTarget
        date=intersect(date,Stockdata{i}(:,1));
    end
    data=[];
    for i=1:nTarget
        [~,~,index]=intersect(date,Stockdata{i}(:,1));
        EachData=Stockdata{i}(index, 5);  %only close price
        data=[data,EachData];
    end
    MatrixNum=size(data,2);
    Days=30;
    for i=1:MatrixNum
        maxClose(i)=max(data(:,i));
        before=data(1:size(data,1)-1,i);
        after=data(2:size(data,1),i);
        Closebefore(i,:)=before./maxClose(i);
        Closeafter(i,:)=after./maxClose(i);
        Gainse{i}=Closeafter(i,:)-Closebefore(i,:);
        
        Close1{i}=before;
        Close2{i}=after;
    end
    %% DataMatrix
    for i=1:MatrixNum
        tmp=1;
        for k=1:length(Gainse{i})
            DataMatrix{i}(k,:)=Gainse{i}(tmp:tmp+Days);
            tmp=tmp+1;
            if tmp+Days>length(Gainse{i})
                break;
            end
        end
    end
    nTrain=ceil(0.83*size(DataMatrix{1},1));
    %% Divide features and targets
    AllFeature=[]; AllTarget=[];
    Feature=[]; Target=[];
    MultiDataMatrix=[];
    for i=1:MatrixNum
        Feature{i}=DataMatrix{i}(:,1:Days);
        Target{i}=DataMatrix{i}(:,Days+1);
        AllFeature=[AllFeature Feature{i}];
        AllTarget=[AllTarget Target{i}];
    end
    temp=1;
    for i=5:5:MatrixNum
        Target{temp}=DataMatrix{i-1}(:,Days+1);
        AllTarget=[AllTarget Target{temp}];
        temp=temp+1;
    end
    MultiDataMatrix=[AllFeature AllTarget];
    %% Influence Matrix
    TotalIIM=InfluenceMatrix(MultiDataMatrix);
    for i=1:nTarget
        FeatureIIM=TotalIIM(1:size(AllFeature,2),1:size(AllFeature,2));
        temp = size(FeatureIIM, 1);
        IIM{i}=[[FeatureIIM; TotalIIM(temp+i, 1:temp)], [TotalIIM(1:temp, temp+i); 0]];
    end
    %% Feature Selection
    FP=FeatureSelection(IIM,nTarget,nFeature,Selection);
    %% Plot Pd
    nFP=size(FP,2);
    tg=[];
    for t=1:nTarget
        fig=figure();
        for i=1:2
            tg(:,t)=AllTarget(:,t);
            [pdt,dt,xt]=Pd(tg(:,t));
            plot(xt,pdt);
            
            title([filename(t,2),'Probability density distributions of features and targets']);
            xlabel('X');
            ylabel('Probability');
            hold on;
            
            nft=MultiDataMatrix(MultiDataMatrix(:,t+(Days*nTarget))<0,i);
            pft=MultiDataMatrix(MultiDataMatrix(:,t+(Days*nTarget))>=0,i);
            [pdnft,dnft,xnft]=Pd(nft);
            [pdpft,dpft,xpft]=Pd(pft);
            plot(xnft,pdnft);
            hold on;
            plot(xpft,pdpft);
            hold on;
        end
        legend(['p_d(t_{',filename{t,2},'})'],['p_d(t_{',filename{t,2},'} | f^-_{',int2str(FP(1)),'})'],['p_d(t_{',filename{t,2},'} | f^+_{',int2str(FP(1)),'})'],['p_d(t_{',filename{t,2},'} | f^-_{',int2str(FP(2)),'})'],['p_d(t_{',filename{t,2},'} | f^+_{',int2str(FP(2)),'})']);
        saveas(fig,['Ex5_2018_',filename{t,2},'_f',int2str(FP(1)),'f',int2str(FP(2)),'_pd.fig']);
    end
    
    hold off;
    %% Experiment
    for p=1:execute
        nDim=size(FP,2);
        for i=1:nDim
            Input(i).value=MultiDataMatrix(:,FP(i))';
            Train(i).value=Input(i).value(1:nTrain);
            Test(i).value=Input(i).value(nTrain+1:end);
            [Train(i).core,Train(i).sigma]=subclust(Train(i).value',0.2);
        end
        %% ContructMatrix
        AC=['ConIndex = allcomb(1:length(Train(1).core)'];
        for i=2:nDim
            AC=[AC,',1:length(Train (',int2str(i),').core)'];
        end
        AC=[AC,');'];
        eval(AC);
        %newC=ConIndex;
        newC=ConstructMatrix(ConIndex,Train);
        %% WOA Initialization
        SearchAgents=30;
        %% ABC Initialization
        SN=30;
        MCN=Iteration;
        Onlooker=SN;
        limit=20;
        %% CACO Initialization
        opts.iteration = Iteration;
        opts.nDim = size(Test,2);
        opts.nAnt = 30;
        opts.new_nAnt = 30;
        opts.eva_rate = 0.9;
        opts.learning_rate = 0.1;
        %% SLPSO
        num_baseParticle=30;
        %% Algorithm
        switch z
            case 1
                [Leader,Convergence_curve,y_hat,TestCost]=WOARLSE(SearchAgents,newC,Iteration,Test,Train,Target,maxClose,Close1,Days);
            case 2
                [Leader,Convergence_curve,y_hat,TestCost]=WOA(SearchAgents,newC,Iteration,Test,Train,Target,maxClose,Close1,Days);
            case 3
                [Leader,Convergence_curve,y_hat,TestCost]=ABCRLSE(SN,MCN,Onlooker,limit,newC,Test,Train,Target,maxClose,Close1,Days);
            case 4
                [Leader,Convergence_curve,y_hat,TestCost]=ABC(SN,MCN,Onlooker,limit,newC,Test,Train,Target,maxClose,Close1,Days);
            case 5
                [Leader,Convergence_curve,y_hat,TestCost]=CACORLSE(opts,newC,Test,Train,Target,maxClose,Close1,Days);
            case 6
                [Leader,Convergence_curve,y_hat,TestCost]=CACO(opts,newC,Test,Train,Target,maxClose,Close1,Days);
            case 7
                [Leader,Convergence_curve,y_hat,TestCost]=SLPSORLSE(num_baseParticle,Iteration,newC,Test,Train,Target,maxClose,Close1,Days);
            case 8
                [Leader,Convergence_curve,y_hat,TestCost]=SLPSO(num_baseParticle,Iteration,newC,Test,Train,Target,maxClose,Close1,Days);
        end
    
    nOutput=ceil(nTarget/2);
    i=1;
    if nOutput>=1
        for t=1:nOutput
            fig=figure();
            plot(Close2{i});
   
            xlabel('Trading date index');
            ylabel('Close price');
            hold on;
            output{i}=Close1{i}(Days+1:end)+(real(y_hat{t}).*maxClose(i));
            title([filename{i,2},' Stock price Prediction']);
            line([(Days+nTrain),(Days+nTrain)],[(min(output{i})-100),(max(output{i})+100)],'Color','k');
            text(120,max(Close2{i}),'Training'); 
            text(220,max(Close2{i}),'Testing'); 
            fitness{i}=Close2{i}(Days+1:length(Close2{i}))-output{i};
            plot(Days+1:length(Close2{i}),output{i},'r--');
            legend('Target function','Proposed approach');
            hold off;
            saveas(fig,['Ex5_2018_',filename{i,2},'_',algorithm{z},'_Stock price Prediction_',int2str(p),'.fig']);
            
            fig=figure();
            plot(Close2{i+1});
            title([filename{i+1,2},' Stock price Prediction']);
            xlabel('Trading date index');
            ylabel('Close price');
            hold on;
            output{i+1}=Close1{i+1}(31:end)+(imag(y_hat{t}).*maxClose(i+1));
             line([(Days+nTrain),(Days+nTrain)],[(min(output{i+1})-100),(max(output{i+1})+100)],'Color','k');
            text(120,max(Close2{i+1}),'Training'); 
            text(220,max(Close2{i+1}),'Testing'); 
            fitness{i+1}=Close2{i+1}(31:length(Close2{i+1}))-output{i+1};
            plot(31:length(Close2{i+1}),output{i+1},'r--');
            legend('Target function','Proposed approach');
            hold off;
            saveas(fig,['Ex5_2018_',filename{i+1,2},'_',algorithm{z},'_Stock price Prediction_',int2str(p),'.fig']);
            i=i+2;
        end
    end
    for i=1:size(fitness,2)
        fig=figure();
        plot(Days+1:length(Close2{i}),fitness{i});
        title([filename{i,2},'Prediction error']);
         line([(Days+nTrain),(Days+nTrain)],[(min(fitness{i})-50),(max(fitness{i})+50)],'Color','k');
        text(120,max(fitness{i}),'Training'); 
        text(220,max(fitness{i}),'Testing'); 
        grid on
        xlabel('Trading date index');
        ylabel('Prediction error');
        saveas(fig,['Ex5_2018_',filename{i,2},'_',algorithm{z},'_Prediction error_',int2str(p),'.fig']);
    end
    for i=1:size(fitness,2)
         StockRMSE{1,i}=filename{i,2};
        StockRMSE{2,i}=RMSE(fitness{i}',size(fitness{i},1))
    end
    fig=figure();
    plot(Convergence_curve);
    xlabel('Iterations');
    ylabel('Cost (RMSE)');
    saveas(fig,['Ex5_2018_',algorithm{z},'_Learning curve_',int2str(p),'.fig']);   
    save(['Ex5_2018_',algorithm{z},'_4Targets_Workspace_',int2str(p)]);
    end
end