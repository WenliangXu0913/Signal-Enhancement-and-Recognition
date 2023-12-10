clear;clc
Fs=44100*0.8;
T_test=0.1;
window=sqrt(hann(128));    
%信号分帧加正弦窗，还原信号再加正弦窗，分帧信号错位相加，避免信号连接处出现畸变。
nfft=128;
noverlap=length(window)/2;
SNRin=[-5,-10,-15,-20]; 
modelname='DnCNN-MIX-64-[-5-5-20]-085-50';
[testinput,S_input,testlabel,S_label,sequence,Fc,Tc,Roottraindata,length_n]=gettestdata(window,noverlap,nfft,Fs,SNRin);
load([modelname,'.mat']);
testpredict=zeros(size(testinput,1),size(testinput,2),size(testinput,3),size(testinput,4));
for i = 1:1:size(testinput,4)
    testpredict(:,:,1,i)=predict(net,testinput(:,:,1,i));
end
%%
Signalcell=cell(2,size(testpredict,4));
for i = 1:1:size(Signalcell,2)
    Signalcell{1,i}=getoriginsignal(S_input(:,:,1,i),testpredict(:,:,1,i),window,noverlap,nfft);
    Signalcell{2,i}=getoriginsignal(S_label(:,:,1,i),testlabel(:,:,1,i),window,noverlap,nfft);
end
%%
finalsignal=cell(2,size(sequence,2));
frame_len=length(Signalcell{1,1});
step_len=frame_len/2;
s1=1;
for i = 1:1:size(finalsignal,2)
    ifinalsignal=zeros((0.5*frame_len*sequence{1,i}(4)+0.5*frame_len),1);
    iSignalcell=cell(1,sequence{i}(4));
    for k = 1:1:sequence{i}(4)
        iSignalcell{1,k}=Signalcell{1,s1+k-1};
    end
    ifinalsignal(1:frame_len-nfft/2)=iSignalcell{1,1}(1:end-nfft/2);
    index=frame_len-nfft/2-(frame_len/2-nfft)+1;
    for j = 2:1:size(iSignalcell,2)
        isignal=iSignalcell{1,j}(nfft/2+1:end-nfft/2);
        ifinalsignal(index:index+(frame_len/2-nfft)-1)=(ifinalsignal(index:index+(frame_len/2-nfft)-1)+isignal(1:(frame_len/2-nfft)))/2;
        ifinalsignal((index+(frame_len/2-nfft)):(index+(frame_len/2-nfft)+length(isignal)-(frame_len/2-nfft)-1))=isignal((frame_len/2-nfft+1):end);
        index=index+step_len;
    end 
    ifinalsignal(end-nfft/2+1:end)=iSignalcell{1,size(iSignalcell,2)}(end-nfft/2+1:end);
    finalsignal{1,i}=ifinalsignal;
    s1=s1+sequence{i}(4);
end
s1=1;
for i = 1:1:size(finalsignal,2)
    ifinalsignal=zeros((0.5*frame_len*sequence{1,i}(4)+0.5*frame_len),1);
    iSignalcell=cell(1,sequence{i}(4));
    for k = 1:1:sequence{i}(4)
        iSignalcell{1,k}=Signalcell{2,s1+k-1};
    end
    ifinalsignal(1:frame_len-nfft/2)=iSignalcell{1,1}(1:end-nfft/2);
    index=frame_len-nfft/2-(frame_len/2-nfft)+1;
    for j = 2:1:size(iSignalcell,2)
        isignal=iSignalcell{1,j}(nfft/2+1:end-nfft/2);
        ifinalsignal(index:index+(frame_len/2-nfft)-1)=(ifinalsignal(index:index+(frame_len/2-nfft)-1)+isignal(1:(frame_len/2-nfft)))/2;
        ifinalsignal((index+(frame_len/2-nfft)):(index+(frame_len/2-nfft)+length(isignal)-(frame_len/2-nfft)-1))=isignal((frame_len/2-nfft+1):end);
        index=index+step_len;
    end 
    ifinalsignal(end-nfft/2+1:end)=iSignalcell{1,size(iSignalcell,2)}(end-nfft/2+1:end);
    finalsignal{2,i}=ifinalsignal;
    s1=s1+sequence{i}(4);
end
%%
SNRout=zeros(1,size(finalsignal,2));
MSE=zeros(1,size(finalsignal,2));
PSNRin=zeros(1,size(finalsignal,2));
for i = 1:1:size(SNRout,2)
    SNRout(1,i)=20*log10(norm(finalsignal{2,i},2)/(norm(finalsignal{2,i}-finalsignal{1,i},2)));
    MSE(1,i)=norm(finalsignal{2,i}-finalsignal{1,i},2)/length(finalsignal{1,i});
    PSNRin(1,i)=10*log10((max(finalsignal{2,i}))^2/MSE(1,i));
end
%%
%Yn带噪；Y增强；Y0纯净
Yn_SNR_cell=cell(1,length_n(3));
Y_SNR_cell=cell(1,length_n(3));
for i = 1:1:length_n(3)
    Yn_signal_cell=cell(length_n(1),1);
    Y_signal_cell=cell(length_n(1),1);
    for j = 1:1:length_n(1)
        Yn_noise_cell=cell(length_n(2),1);
        Y_noise_cell=cell(length_n(2),1);
        for k = 1:1:length_n(2)
            for ii = 1:1:size(sequence,2)
                if sequence{ii}(1)==j && sequence{ii}(2)==k && sequence{ii}(3)==i
                    Yn_noise_cell{k}=Roottraindata{1,ii};
                    Y_noise_cell{k}=finalsignal{1,ii};
                end
            end
        end
        Yn_signal_cell{j}=Yn_noise_cell;
        Y_signal_cell{j}=Y_noise_cell;
    end
    Yn_SNR_cell{i}=Yn_signal_cell;
    Y_SNR_cell{i}=Y_signal_cell;
end
Y0_SNR_cell=cell(1,length_n(1));
for i = 1:1:length_n(1)
    Y0_SNR_cell{i}=Roottraindata{2,1+(i-1)*length_n(2)*length_n(3)};
end
%训练模型
Y0_train=cell(size(Y0_SNR_cell,2),1);
Y0_label=cell(size(Y0_SNR_cell,2),1);
for i = 1:1:size(Y0_SNR_cell,2)
    Y0_train{i}=getsymbolfortest(Y0_SNR_cell{i},T_test,Fs);
    Y0_label{i}=ones(size(Y0_train{i},1),1)*i;
end
Y0_train=cell2mat(Y0_train);
Y0_label=cell2mat(Y0_label);
% n=randperm(size(Y0_train,1));
load('n5.mat');
%%
%训练集样本集
train_features=Y0_train(n,:);
train_label=Y0_label(n,:);
%归一化处理
[train_feturesC,PS]=mapminmax(train_features');
train_features=train_feturesC';
%创建训练模型
[bestacc,bestc,bestg]=SVMcgForClass(train_label,train_features,-10,5,-10,5,3,0.5,0.5);
%注意空格，必须要有
cmd=['-c ',num2str(bestc),' -g ',num2str(bestg)];
% cmd=['-c ',num2str(0.125),' -g ',num2str(0.3536)];
SVM_model=libsvmtrain(train_label,train_features,cmd);
% SVM_model=libsvmtrain(train_label,train_features);
[predict_train_label]=libsvmpredict(train_label,train_features,SVM_model);
compare_train=(train_label==predict_train_label);
accuracy_train=sum(compare_train)/size(train_label,1)*100;
fprintf('纯净信号模型训练准确率：%f\n',accuracy_train);
%%
Yn_accuracy_test=zeros(1,length_n(3));
for i = 1:1:length_n(3)
    Yn_test=cell(length_n(1),1);
    Yn_label=cell(length_n(1),1);
    for j = 1:1:length_n(1)
        Yn_test_sub=cell(length_n(2),1);
        Yn_label_sub=cell(length_n(2),1);
        for k = 1:1:length_n(2)
            Yn_test_sub{k}=getsymbolfortest(Yn_SNR_cell{1,i}{j,1}{k,1},T_test,Fs);
            Yn_label_sub{k}=ones(size(Yn_test_sub{k},1),1)*j;
        end
        Yn_test{j}=cell2mat(Yn_test_sub);
        Yn_label{j}=cell2mat(Yn_label_sub);
    end
    Yn_test_features=cell2mat(Yn_test);
    Yn_test_features=mapminmax('apply',Yn_test_features',PS)';
    Yn_test_labels=cell2mat(Yn_label);
    [predict_test_label]=libsvmpredict(Yn_test_labels,Yn_test_features,SVM_model);
    compare_test=(Yn_test_labels==predict_test_label);
    Yn_accuracy_test(i)=sum(compare_test)/size(Yn_test_labels,1)*100;
    fprintf('SNRIN：%d；带噪信号测试准确率：%f\n',[SNRin(i),Yn_accuracy_test(i)]);
end

Y_accuracy_test=zeros(1,length_n(3));
for i = 1:1:length_n(3)
    Y_test=cell(length_n(1),1);
    Y_label=cell(length_n(1),1);
    for j = 1:1:length_n(1)
        Y_test_sub=cell(length_n(2),1);
        Y_label_sub=cell(length_n(2),1);
        for k = 1:1:length_n(2)
            Y_test_sub{k}=getsymbolfortest(Y_SNR_cell{1,i}{j,1}{k,1},T_test,Fs);
            Y_label_sub{k}=ones(size(Y_test_sub{k},1),1)*j;
        end
        Y_test{j}=cell2mat(Y_test_sub);
        Y_label{j}=cell2mat(Y_label_sub);
    end
    Y_test_features=cell2mat(Y_test);
    Y_test_features=mapminmax('apply',Y_test_features',PS)';
%     Y_test_features=mapminmax(Y_test_features')';
    Y_test_labels=cell2mat(Y_label);
    [predict_test_label]=libsvmpredict(Y_test_labels,Y_test_features,SVM_model);
    compare_test=(Y_test_labels==predict_test_label);
    Y_accuracy_test(i)=sum(compare_test)/size(Y_test_labels,1)*100;
    fprintf('SNRIN：%d；增强信号测试准确率：%f\n',[SNRin(i),Y_accuracy_test(i)]);
end
%%
% periodogram(finalsignal{1,4},rectwin(length(finalsignal{1,4})),length(finalsignal{1,4}),Fs);
% pwelch(finalsignal{1,1},hamming(16384),8192,1024,44100);
% hold on;
% pwelch(finalsignal{2,1},hamming(16384),8192,1024,44100);
%%
function [input,S_input,label,S_label,sequence,Fc,Tc,Roottraindata,length_n]=gettestdata(window,noverlap,nfft,Fs,SNRin)
    % 取出所有原始数据
    folder = {'TestWaternoise','TestPuresignal'};
    ext = {'*.TXT'};
    Noisefilepaths = [];
    Signalfilepaths = [];
    for i = 1 : length(ext)
        Noisefilepaths = cat(1,Noisefilepaths, dir(fullfile(folder{1}, ext{i})));
    end
    Noisedata=cell(1,length(Noisefilepaths));
    for i = 1:1:length(Noisefilepaths)
        Noisedata{i}=importdata(fullfile(folder{1},Noisefilepaths(i).name));
    end
    for i = 1 : length(ext)
        Signalfilepaths = cat(1,Signalfilepaths, dir(fullfile(folder{2}, ext{i})));
    end
    Signaldata=cell(1,length(Signalfilepaths));
    for i = 1:1:length(Signalfilepaths)
        Signaldata{i}=importdata(fullfile(folder{2},Signalfilepaths(i).name));
    end
    Roottraindata=cell(2,length(Signaldata)*length(Noisedata)*length(SNRin));
    sequence=cell(1,length(Signaldata)*length(Noisedata)*length(SNRin));
    length_n=[length(Signaldata),length(Noisedata),length(SNRin)];
    index=1;
    for i = 1:1:length(Signaldata)
        for j = 1:1:length(Noisedata)
            for k = 1:1:length(SNRin)
                
                Cavitation=Signaldata{i};
                Wnoise=Noisedata{j};
                if length(Cavitation)>length(Wnoise)
                    if mod(length(Wnoise),2)==1
                        Cavitation=Cavitation(1:length(Wnoise)-1);
                        Wnoise=Wnoise(1:length(Cavitation));
                    else 
                        Cavitation=Cavitation(1:length(Wnoise));
                        Wnoise=Wnoise(1:length(Cavitation));
                    end
                else
                    if mod(length(Cavitation),2)==1
                        Wnoise=Wnoise(1:length(Cavitation)-1);
                        Cavitation=Cavitation(1:length(Wnoise));
                    else
                        Wnoise=Wnoise(1:length(Cavitation));
                        Cavitation=Cavitation(1:length(Wnoise));
                    end
                end
                Roottraindata{1,index}=getSNRin(Cavitation,Wnoise,SNRin(k));
                Roottraindata{2,index}=Cavitation;
                sequence{index}=[i,j,k];
                index=index+1;
            
            end
        end
    end
    
    % 分帧
    midP=nfft/2+1;
    cult_len=midP*noverlap+(length(window)-noverlap);
    cult_step=cult_len/2;
    cult_sum=zeros(1,size(Roottraindata,2));
    for i = 1:1:size(Roottraindata,2)
        cult_sum(i)=fix((length(Roottraindata{1,i})-(cult_len-cult_step))/cult_step);
        sequence{i}=[sequence{i},cult_sum(i)];
    end
    cult_cell=cell(2,sum(cult_sum));
    index=1;
    for i = 1:1:size(Roottraindata,2)
        indexC=1;
        for j = 1:1:cult_sum(i)
            cult_cell{1,index}=Roottraindata{1,i}(indexC:indexC+cult_len-1);
            cult_cell{2,index}=Roottraindata{2,i}(indexC:indexC+cult_len-1);
            indexC=indexC+cult_step;
            index=index+1;
        end
    end
    
    % 生成训练数据
    input=zeros(midP,midP,1,size(cult_cell,2));
    S_input=zeros(midP,midP,1,size(cult_cell,2));
    label=zeros(midP,midP,1,size(cult_cell,2));
    S_label=zeros(midP,midP,1,size(cult_cell,2));
    Fc=zeros(midP,1,1,size(cult_cell,2));
    Tc=zeros(1,midP,1,size(cult_cell,2));
    k=zeros(midP,1);
    k(2:end-1)=2/(Fs*(window'*window));
    k([1,end])=1/(Fs*(window'*window));
    for i = 1:1:size(cult_cell,2)
        [S,F,T,P]=spectrogram(cult_cell{1,i},window,noverlap,nfft,Fs);
        for j = 1:1:size(P,2)
            P(:,j)=P(:,j)./k;
        end
        input(:,:,1,i)=log10(P);
        S_input(:,:,1,i)=S;
        Fc(:,:,1,i)=F;
        Tc(:,:,1,i)=T;
    end
    for i = 1:1:size(cult_cell,2)
        [S,~,~,P]=spectrogram(cult_cell{2,i},window,noverlap,nfft,Fs);
        for j = 1:1:size(P,2)
            P(:,j)=P(:,j)./k;
        end
        label(:,:,1,i)=log10(P);
        S_label(:,:,1,i)=S;
    end
    

    function SNRinSignal=getSNRin(Signal,Noise,SNRin)
        scalingfactor=norm(Signal,2)/norm(Noise,2)*10^(-(SNRin/20));
        SNRinSignal=Signal+scalingfactor*Noise;
    end
end
function origin_signal=getoriginsignal(S,P,window,noverlap,nfft)
    midP=nfft/2+1;
    frame=size(S,2);
    S_len=zeros(nfft,frame);
    theta_len=zeros(nfft,frame);
    P_len=zeros(nfft,frame);
    Fsignal=zeros(nfft,frame);
%     k=zeros(midP,1);
%     k(2:end-1)=2/(Fs*(window'*window));
%     k([1,end])=1/(Fs*(window'*window));
    for j=1:1:frame
        S_len(1:midP,j)=S(:,j);
        S_len(midP+1:end,j)=flipud(conj(S_len(2:midP-1,j)));
        theta_len(:,j)=angle(S_len(:,j));
        P_len(1:midP,j)=10.^P(:,j);                         %对数还原
        P_len(midP+1:end,j)=flipud(P_len(2:midP-1,j));
        Fsignal(:,j)=real(ifft(sqrt(P_len(:,j)).*(cos(theta_len(:,j))+sqrt(-1)*sin(theta_len(:,j))))).*window;
    end
    index=1;
    origin_signal=zeros((frame*nfft/2+nfft/2),1);
    for j=1:1:frame
        sframe=Fsignal(:,j);
        origin_signal(index:index+nfft-1)=origin_signal(index:index+nfft-1)+sframe(1:end);
        index=index+noverlap;
    end
end

function symbol_test=getsymbolfortest(data,T_test,Fs)
%     data=mapminmax(data')';
%     data=zscore(data);
    nn_test=T_test*Fs;
    L=length(data);
    n_test=floor(L/nn_test);
    dataarry3d_0=zeros(nn_test,1);
    dataarry3d_test=zeros(nn_test,1,n_test);
    for i=1:1:n_test
        dataarry3d_0(:,1)=data((i-1)*nn_test+1:i*nn_test,1);
        dataarry3d_test(:,1,i)=dataarry3d_0(:,1);
    end

    ma=zeros(n_test,1);               % 每一条通道的数据求一个最大值
    mi=zeros(n_test,1);               % 每一条通道的数据求一个最小值
    doubleamp=zeros(n_test,1);        % 每一条通道的数据求一个双幅值
    m=zeros(n_test,1);                % 每一条通道的数据求一个均值
    av=zeros(n_test,1);               % 每一条通道的数据求一个整流平均值
    stdz=zeros(n_test,1);             % 每一条通道的数据求一个标准差
    va=zeros(n_test,1);               % 每一条通道的数据求一个方差
    ku=zeros(n_test,1);               % 每一条通道的数据求一个峭度
    sk=zeros(n_test,1);               % 每一条通道的数据求一个偏斜度
    rm=zeros(n_test,1);               % 每一条通道的数据求一个均方根
    waveform=zeros(n_test,1);         % 每一条通道的数据求一个波形因子
    peakfactor=zeros(n_test,1);       % 每一条通道的数据求一个峰值因子
    pulsefactor=zeros(n_test,1);      % 每一条通道的数据求一个脉冲因子
    FC=zeros(n_test,1);               % 每一条通道的数据求一个重心频率
    MSF=zeros(n_test,1);              % 每一条通道的数据求一个均方频率
    RMSF=zeros(n_test,1);             % 每一条通道的数据求一个均方根频率
    VF=zeros(n_test,1);               % 每一条通道的数据求一个频率方差
    RVF=zeros(n_test,1);              % 每一条通道的数据求一个频率标准差
    clearance=zeros(n_test,1);        % 每一条通道的数据求一个裕度因子
    fmax=zeros(n_test,1);             % 每一条通道的数据求一个卓越频率
    psdE=zeros(n_test,1);             % 每一条通道的数据求一个功率谱熵
    eE=zeros(n_test,1);               % 每一条通道的数据求一个能量熵

    for i=1:1:1
        for j=1:1:n_test
            m(j,i)=mean(dataarry3d_test(:,i,j));                                        % 求均值去除直流分量
            dataarry3d_test(:,i,j)=dataarry3d_test(:,i,j)-m(j,i);
            ma(j,i)=max(dataarry3d_test(:,i,j));                                        % 求最大值
            mi(j,i)=min(dataarry3d_test(:,i,j));                                        % 求最小值
            ku(j,i)=kurtosis(dataarry3d_test(:,i,j));                                   % 求峭度
            sk(j,i)=skewness(dataarry3d_test(:,i,j));                                   % 求偏斜度
            doubleamp(j,i)=max(dataarry3d_test(:,i,j))-min(dataarry3d_test(:,i,j));     % 双幅值
            stdz(j,i)=std(dataarry3d_test(:,i,j));                                      % 标准差
            av(j,i)=mean(abs(dataarry3d_test(:,i,j)));                                  % 求整流平均值
            va(j,i)=var(dataarry3d_test(:,i,j));                                        % 求方差
            rm(j,i)=rms(dataarry3d_test(:,i,j));                                        % 求均方根
            waveform(j,i)=rm(j,i)/av(j,i);                                              % 求波形因子
            peakfactor(j,i)=doubleamp(j,i)/rm(j,i);                                     % 求峰值因子
            pulsefactor(j,i)=doubleamp(j,i)/av(j,i);                                    % 求脉冲因子
            [p,f]=periodogram(dataarry3d_test(:,i,j),[],[],Fs);                         %功率谱，p为功率谱幅值，f为频率轴
            FC(j,i)=sum(p.*f)./sum(p);                                                  %求重心频率
            MSF(j,i)=sum(f.^2.*p)./sum(p);                                              %求均方频率
            RMSF(j,i)= sqrt(MSF(j,i));                                                  %求均方根频率
            VF(j,i)=sum((f-FC(j,i)).^2.*p)./sum(p);                                     %求频率方差
            RVF(j,i)=sqrt(VF(j,i));                                                     %求频率标准差
            clearance(j,i)=doubleamp(j,i)/mean(sqrt(abs(dataarry3d_test(:,i,j))))^2;    % 求裕度因子   
            f=linspace(0,Fs/2,nn_test/2);                                               % 频率序列
            M=fft(dataarry3d_test(:,i,j))/nn_test;                                      % 快速傅立叶变换
            fz=(2*abs(M)').^2;                                                          % 转换为功率谱
            [~,ind]=max(fz(1:nn_test/2));                                               % 求最大值和对应频率坐标
            fmax(j,i)=f(ind);                                                           % 卓越频率
            psdE(j,i)=kPowerSpectrumEntropy(dataarry3d_test(:,i,j));                    % 求功率谱熵
            eE(j,i)=kEnergyEntropy(dataarry3d_test(:,i,j));                             % 求能量熵
        end
    end

    symbol_test=zeros(n_test,22,1);
    for i=1:1:1
        symbol_test(:,:,i)=[ma(:,i),mi(:,i),doubleamp(:,i),m(:,i),...
        stdz(:,i),av(:,i),va(:,i),ku(:,i),sk(:,i),rm(:,i),...
        waveform(:,i),peakfactor(:,i),pulsefactor(:,i),FC(:,i),...
        MSF(:,i),RMSF(:,i),VF(:,i),RVF(:,i),clearance(:,i),fmax(:,i),...
        psdE(:,i),eE(:,i)];
    end
end

