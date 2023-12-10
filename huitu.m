i=1;
y0=Roottraindata{2,i};
y0=y0(35281:70560);
yn=Roottraindata{1,i};
yn=yn(35281:70560);
y=finalsignal{1,i};
y=y(35281:70560);
%%
% [~,f,t,p]=spectrogram(yn,window,noverlap,1024,Fs,'power');
% [~,f,t,p]=spectrogram(y0,window,noverlap,1024,Fs,'power');
% spec=zeros(size(p)+1);
% spec(1,2:end)=t;
% spec(2:end,1)=f;
% spec(2:end,2:end)=10*log10(abs(p));

% ax = gca;
% ax.YDir = 'reverse';

%%
pwelch(y,hamming(16384),8192,1024,35280);
hold on;
ax = gca;
ax.LineStyleOrder={'-','-.',':'};
ax.ColorOrder=[0.00 0.45 0.74;0.85 0.33 0.10;0.36 0.52 0.15];
ax.LineStyleOrderIndex=2;
ax.ColorOrderIndex=2;
pwelch(y0,hamming(16384),8192,1024,35280);
ax.LineStyleOrderIndex=3;
ax.ColorOrderIndex=3;
pwelch(yn,hamming(16384),8192,1024,35280);
legend('增强信号','纯净信号','带噪信号','Location','northwest');grid off;title('');
hold off;
% %%
% obj=get(gca,'children');
% axisdata=[];
% for p = 1:1:3
%     xd=get(obj(p),'xdata');
%     yd=smooth(get(obj(p),'ydata'),15)';
%     axisdataC=[xd',yd'];
%     axisdata=[axisdata,axisdataC];
% end
% writematrix(axisdata,'axisdatasmooth.xlsx','Sheet',i);