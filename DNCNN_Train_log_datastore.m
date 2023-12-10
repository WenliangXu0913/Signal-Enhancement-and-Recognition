clear;clc
Fs=44100*0.8;
window=sqrt(hann(128));    
%信号分帧加正弦窗，还原信号再加正弦窗，分帧信号错位相加，避免信号连接处出现畸变。
nfft=128;
noverlap=length(window)/2;
SubTrainDataset='SubTrainDataset';
SubValidateDataset='SubValidateDataset';
%%
cult_cell=getcult_cell(window,noverlap,nfft);
n = randperm(size(cult_cell,2));
cult_cell_train=cult_cell(:,n(1:floor(length(n)*0.8)));
cult_cell_validate=cult_cell(:,n(floor(length(n)*0.8)+1:end));

if ~isfolder(SubTrainDataset)
    mkdir(SubTrainDataset);
    for i=1:1:size(cult_cell_train,2)
        filename=['SubTrainDataset/',num2str(i),'.txt'];
        fid=fopen(filename,"w");
        fprintf(fid,'%10.5f   %10.5f\r\n',[cult_cell_train{:,i}]');
        fclose(fid);
    end
end

if ~isfolder(SubValidateDataset)
    mkdir(SubValidateDataset);
    for i=1:1:size(cult_cell_validate,2)
        filename=['SubValidateDataset/',num2str(i),'.txt'];
        fid=fopen(filename,"w");
        fprintf(fid,'%10.5f   %10.5f\r\n',[cult_cell_validate{:,i}]');
        fclose(fid);
    end
end
%%
SNRlimitation=[0,-40];
dsTrain=createFileDatastore(SubTrainDataset,SNRlimitation);
dsValidate=createFileDatastore(SubValidateDataset,SNRlimitation);
dsTrain=transform(dsTrain,@(data)reprocessdata(data,window,noverlap,nfft,Fs));
dsValidate=transform(dsValidate,@(data)reprocessdata(data,window,noverlap,nfft,Fs));
%%
options=trainingOptions("adam",...
    'MiniBatchSize',64,...
    'InitialLearnRate',0.001,...
    'LearnRateSchedule',"piecewise",...
    'LearnRateDropPeriod',2,...
    'LearnRateDropFactor',0.85,...
    'MaxEpochs',100,...
    'ValidationFrequency',5000,...
    'ValidationData',dsValidate,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu',...
    'Plots','training-progress');
layers = creatlayers_log_res4_ceshi3;
net = trainNetwork(dsTrain,layers,options);
save('DnCNN-DS-[0-40]-mix','net');
%%
function cult_cell=getcult_cell(window,noverlap,nfft)
    % 取出所有原始数据
    folder = {'Waternoise','Puresignal'};
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
    Roottraindata=cell(2,length(Signaldata)*length(Noisedata));
    index=1;
    for i = 1:1:length(Signaldata)
        for j = 1:1:length(Noisedata)
                
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
                Roottraindata{1,index}=Wnoise;
                Roottraindata{2,index}=Cavitation;
                index=index+1;
            
        end
    end
    
    % 分帧
    midP=nfft/2+1;
    cult_len=midP*noverlap+(length(window)-noverlap);
    cult_step=cult_len/2;
    cult_sum=zeros(1,size(Roottraindata,2));
    for i = 1:1:size(Roottraindata,2)
        cult_sum(i)=fix((length(Roottraindata{1,i})-(cult_len-cult_step))/cult_step);
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
end

function datastore=createFileDatastore(fileloc,SNRlimitation)
    readFcn=@(f)readData(f,SNRlimitation);
    datastore=fileDatastore(fileloc,...
        "IncludeSubfolders",true,...
        "FileExtensions",'.txt',...
        'ReadFcn',readFcn,...
        'ReadMode','file');
end

function data=readData(filename,SNRlimitation)
    fid=fopen(filename);
    rdata=textscan(fid,'%10.5f   %10.5f');
    fclose(fid);
    wndata=rdata{1,1};
    sidata=rdata{1,2};
    yndata=getSNRin(sidata,wndata,(-randi(-SNRlimitation)));
    data={yndata,sidata};
end

function SNRinSignal=getSNRin(Signal,Noise,SNRin)
    scalingfactor=norm(Signal,2)/norm(Noise,2)*10^(-(SNRin/20));
    SNRinSignal=Signal+scalingfactor*Noise;
end

function data=reprocessdata(data,window,noverlap,nfft,Fs)
    midP=nfft/2+1;
    input=zeros(midP,midP);
    label=zeros(midP,midP);
    k=zeros(midP,1);
    k(2:end-1)=2/(Fs*(window'*window));
    k([1,end])=1/(Fs*(window'*window));

    [~,~,~,P]=spectrogram(data{1},window,noverlap,nfft,Fs);
    for j = 1:1:size(P,2)
        P(:,j)=P(:,j)./k;
    end
    input(:,:)=log10(P);

    [~,~,~,P]=spectrogram(data{2},window,noverlap,nfft,Fs);
    for j = 1:1:size(P,2)
        P(:,j)=P(:,j)./k;
    end
    label(:,:)=log10(P);

    data={single(input),single(label)};
end