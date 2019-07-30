clear all
close all

ndim=6;

%% Plot 

data=load('trajectory.out');
xlims=[900,1300];

ylims=[ 0.7,  1;
       -0.2,0.4;
       -0.5,0.5;
         -1,  0;
       -0.5,0.5;
       -0.5,  1];
         
for ii=1:ndim;
   figure(ii)
   plot(data(:,1),data(:,ii+1), 'k-')
   xlabel('$t$'  , 'interpreter', 'latex')
   ylbl = ['$z_{',num2str(ii),'}$'];
   ylabel(ylbl, 'interpreter', 'latex')
   xlim(xlims); ylim(ylims(ii,:))
   xticks([900,1100,1300]);
   set(gca, 'FontSize', 8)
   hAxes=gca;
   hAxes.TickLabelInterpreter = 'latex';

   fig = gcf;
   fig.PaperUnits = 'inches';
   fig.PaperPosition = [0 0 2 1.3];
   print(['traj',num2str(ii)],'-depsc')
end

