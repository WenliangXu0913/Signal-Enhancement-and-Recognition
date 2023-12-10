function [input,label]=gettraindata_log_MS(window,noverlap,nfft,Fs,SNRin)
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
    
    cellnum=size(cult_cell,2);
    cult_cell=repmat(cult_cell,1,length(SNRin));
    for i=1:1:length(SNRin)
        for index=1:1:cellnum
            cult_cell{1,index+(i-1)*cellnum}=getSNRin(cult_cell{2,index+(i-1)*cellnum},cult_cell{1,index+(i-1)*cellnum},SNRin(i));
        end
    end

    % 生成训练数据
    input=zeros(midP,midP,1,size(cult_cell,2));
    label=zeros(midP,midP,1,size(cult_cell,2));
    k=zeros(midP,1);
    k(2:end-1)=2/(Fs*(window'*window));
    k([1,end])=1/(Fs*(window'*window));
    for i = 1:1:size(cult_cell,2)
        [~,~,~,P]=spectrogram(cult_cell{1,i},window,noverlap,nfft,Fs);
        for j = 1:1:size(P,2)
            P(:,j)=P(:,j)./k;
        end
        input(:,:,1,i)=log10(P);
    end
    for i = 1:1:size(cult_cell,2)
        [~,~,~,P]=spectrogram(cult_cell{2,i},window,noverlap,nfft,Fs);
        for j = 1:1:size(P,2)
            P(:,j)=P(:,j)./k;
        end
        label(:,:,1,i)=log10(P);
    end
    

    function SNRinSignal=getSNRin(Signal,Noise,SNRin)
        scalingfactor=norm(Signal,2)/norm(Noise,2)*10^(-(SNRin/20));
        SNRinSignal=Signal+scalingfactor*Noise;
    end
end