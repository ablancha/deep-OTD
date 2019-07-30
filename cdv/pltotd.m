clear all
close all

notd=1;
ndim=6;


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


%% Plot 
colors = [ 202,0,32;
           5,113,176]/256;

for ii=1:notd;
   figure(ii)
   for jj=1:ndim;
     subplot(ndim,1,jj);
    %sc=sign(dot(uNum(:,:,ii),uDeep(:,:,ii),2)); % Flip if sign difference
     sc=1.0;
     plot(   t,sc.*uDeep(:,jj,ii), '-', 'color', colors(1,:)); hold on
     plot(   t,     uNum(:,jj,ii), '-', 'color', colors(2,:))
     xlim([500,4000]);
     xlabel('$t$'  , 'interpreter', 'latex')
     ylbl=['$u_{',num2str(ii),', z_' num2str(jj), '}$'];
     ylabel(ylbl, 'interpreter', 'latex')
     set(gca, 'FontSize', 8)
     hAxes=gca;
     hAxes.TickLabelInterpreter = 'latex';
   end
  fig = gcf;
  fig.PaperUnits = 'inches';
  fig.PaperPosition = [0 0 6.5 8.0];
  print(['ts',num2str(ii)],'-depsc')
end

