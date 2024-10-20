function demo_Gaussian_conditioning01
% Conditional probability with a multivariate normal distribution.
%
% @article{Calinon16JIST,
% 	author="Calinon, S.",
% 	title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
% 	journal="Intelligent Service Robotics",
%		publisher="Springer Berlin Heidelberg",
%		doi="10.1007/s11370-015-0187-9",
%		year="2016",
%		volume="9",
%		number="1",
%		pages="1--29"
% }
% 
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbVar = 2; %Number of variables [x1,x2]


%% Load  data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/faithful.mat');
Data = faithful';
Data(1,:) = Data(1,:)*1E1;


%% Gaussian conditioning 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.Mu = mean(Data,2);
model.Sigma = cov(Data');

in=1; out=2;
DataIn = 50;

%See for example section "2.3.1 Conditional Gaussian distributions" in Bishop's book, 
%or Conditional distribution section on the multivariate normal distribution wikipedia page
MuOut = model.Mu(out) + model.Sigma(out,in) / model.Sigma(in,in) * (DataIn-model.Mu(in));
SigmaOut = model.Sigma(out,out) - model.Sigma(out,in) / model.Sigma(in,in) * model.Sigma(in,out);
slope = model.Sigma(out,in)/model.Sigma(in,in);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mrg = [10 10];
limAxes = [min(Data(1,:))-mrg(1) max(Data(1,:))+mrg(1) min(Data(2,:))-mrg(2) max(Data(2,:))+mrg(2)];

figure('PaperPosition',[0 0 4 3],'position',[10,50,1600,1200]); hold on; %axis off;
plot(DataIn+[-50,50], MuOut+slope*[-50,50], ':','linewidth',1,'color',[.7 .3 .3]);
plot([model.Mu(1) model.Mu(1)], [limAxes(3) model.Mu(2)], ':','linewidth',1,'color',[.7 .3 .3]);
plot([limAxes(1) model.Mu(1)], [model.Mu(2) model.Mu(2)], ':','linewidth',1,'color',[.7 .3 .3]);

%Plot joint distribution
plotGMM(model.Mu, model.Sigma, [.8 0 0]);
%Plot marginal distribution horizontally
plotGaussian1D(model.Mu(1), model.Sigma(1,1), [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) 10], [.8 0 0], 1, 'h');
%Plot marginal distribution vertically
plotGaussian1D(model.Mu(2), model.Sigma(2,2), [limAxes(1) limAxes(3) 10 limAxes(4)-limAxes(3)], [.8 0 0], 1, 'v');
%Plot conditional distribution vertically
plotGaussian1D(MuOut, SigmaOut, [limAxes(1) limAxes(3) 10 limAxes(4)-limAxes(3)], [0 0 .8], 1, 'v');

plot(DataIn,MuOut,'.','markersize',12,'color',[.7 .3 .3]);
plot(DataIn,limAxes(3),'.','markersize',26,'color',[0 0 .8]);
plot([DataIn DataIn], [limAxes(3) MuOut], ':','linewidth',1,'color',[.7 .3 .3]);
plot([limAxes(1) DataIn], [MuOut MuOut], ':','linewidth',1,'color',[.7 .3 .3]);

axis(limAxes);
set(gca,'Xtick',[]); set(gca,'Ytick',[]);
xlabel('x^I'); ylabel('x^O');

%pause;
%close all;
