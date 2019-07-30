clear all
close all

notd=3;
ndim=3;


%% dOTD modes
for ii=1:notd;
    data=load(['dOTD_testing',num2str(ii),'.out']);
    t=data(:,1);
    if ii==1; uDeep=data(:,2:end); 
    else; uDeep=cat(3,uDeep,data(:,2:end)); end;
end;


%% OTD modes from numerical integration
for ii=1:notd;
    data=load(['OTD_num',num2str(ii),'.out']);
    t=data(:,1);
    if ii==1; uNum=data(:,2:end);
    else; uNum=cat(3,uNum,data(:,2:end)); end;
end;


%% Training stamps
ind_trn=load('training_stmps.txt');
tTrn=t(ind_trn+1);	% +1 because of Python indexing
uTrn=uNum(ind_trn+1,:,:);


%% Plot 
colors = [ 202,0,32;
           5,113,176]/256;

for ii=1:notd;
   figure(ii)
   for jj=1:ndim;
     subplot(ndim,1,jj); 
     sc=sign(dot(uNum(:,:,ii),uDeep(:,:,ii),2)); % Flip if sign difference
     plot(   t,sc.*uDeep(:,jj,ii), '-' , 'color', colors(1,:)); hold on
     plot(   t,     uNum(:,jj,ii), '-.', 'color', colors(2,:))
     plot(tTrn,     uTrn(:,jj,ii), 'k*', 'markersize', 4)
     xlabel('$t-500$'  , 'interpreter', 'latex')
     ylabels = [ ['$u_{',num2str(ii),',x}$'];
                 ['$u_{',num2str(ii),',y}$']; 
                 ['$u_{',num2str(ii),',z}$'];];
     ylabel(ylabels(jj,:), 'interpreter', 'latex')
     xlim([tTrn(1),tTrn(1)+4*pi]); ylim([-1.2,1.2])
     xticks([tTrn(1), tTrn(1)+2*pi, tTrn(1)+4*pi])
     yticks([-1.2,0,1.2])
     xticklabels({'$0$','$2\pi$','$4\pi$'});
     set(gca, 'FontSize', 8)
     hAxes=gca;
     hAxes.TickLabelInterpreter = 'latex';
   end
  fig = gcf;
  fig.PaperUnits = 'inches';
  fig.PaperPosition = [0 0 2.  4.0];
  print(['ts',num2str(ii)],'-depsc')
end

