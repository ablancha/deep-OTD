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
%for ii=1:notd;
%    data=load(['OTD_num',num2str(ii),'.out']);
%    t=data(:,1);
%    if ii==1; uNum=data(:,2:end);
%    else; uNum=cat(3,uNum,data(:,2:end)); end;
%end;

% ODE solution
for ii=1:notd;
    data=load(['myotd',num2str(ii),'.txt']);
    ind=8000:40000-1;
    if ii==1; uNum=data(ind,:);
    else; uNum=cat(3,uNum,data(ind,:)); end;
end;


%% Trajectory
data=load('myFile.txt');
t1=data(:,1);
z1=data(:,2);


%% Plot distance
colors = [ 202,0,32;
           5,113,176]/256;

xlims=[800,2200];

for ii=1:notd;
  figure(ii)

  subplot(2,1,1)
  plot(t1,z1, 'k')
  xlabel('$t$'  , 'interpreter', 'latex')
  ylabel('$z_1$', 'interpreter', 'latex')
  ylim([0.7 1])
  yticks([0.7 1])
  xlim(xlims)
  set(gca, 'FontSize', 8)
  hAxes=gca;
  hAxes.TickLabelInterpreter = 'latex';

  subplot(2,1,2)
  sc=abs(dot(uNum(:,:,ii),uDeep(:,:,ii),2));
  plot(t, sc, '-', 'color', colors(1,:)); hold on;
  xlim(xlims)
  ylim([0, 1]);
  xlabel('$t$'  , 'interpreter', 'latex')
  ylbl=['$d_{',num2str(ii),'}$'];
  ylabel(ylbl, 'interpreter', 'latex')
  set(gca, 'FontSize', 8)
  hAxes=gca;
  hAxes.TickLabelInterpreter = 'latex';

  fig = gcf;
  fig.PaperUnits = 'inches';
  fig.PaperPosition = [0 0 6.4  2.2];
  print(['dist',num2str(ii)],'-depsc')
end


