function demo_Gaussian_lawTotalCov01
% Gaussian estimate of a GMM with the law of total covariance
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
% Copyright (c) 2017 Idiap Research Institute, http://idiap.ch/
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


%% GMM parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1D Gaussians
model.nbVar = 1;
model.nbStates = 3;
model.Mu(1,1:3) = [0, 1.5, 3.5];
model.Sigma(1,1,1:3) = [2, 1, .5];
model.Priors = [.1, .6, .3];

%2D Gaussians
% model.nbVar = 2;
% model.nbStates = 4;
% % model.Priors = rand(1,model.nbStates);
% % model.Priors = model.Priors / sum(model.Priors);
% model.Priors = ones(1,model.nbStates) / model.nbStates;
% model.Mu = rand(model.nbVar,model.nbStates);
% for i=1:model.nbStates
% 	U = rand(model.nbVar,model.nbStates) * 5E-1;
% 	model.Sigma(:,:,i) = U*U';
% end


%% Gaussian estimate of a GMM (law of total variance)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mu = model.Mu * model.Priors';
Sigma = zeros(model.nbVar);
for i=1:model.nbStates
	Sigma = Sigma + model.Priors(i) * (model.Sigma(:,:,i) + model.Mu(:,i)*model.Mu(:,i)');
end
Sigma = Sigma - Mu*Mu';

% %Equivalent computation for 2 Gaussians
% Sigma = model.Priors(1) * model.Sigma(:,:,1) + model.Priors(2) * model.Sigma(:,:,2) + model.Priors(1) * model.Priors(2) * (model.Mu(:,1)-model.Mu(:,2)) * (model.Mu(:,1)-model.Mu(:,2))'


%% Plot 1D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
limAxes = [-3 7 0 1];
figure('PaperPosition',[0 0 5.5 2.475],'position',[10,50,1600,800]); hold on; %axis off;
plotGaussian1D(Mu(1), Sigma(1,1), [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) gaussPDF(0,0,Sigma(1,1))], [.8 0 0], 1, 'h');
Pt = zeros(1,100);
for i=1:model.nbStates
	Ptmp = plotGaussian1D(model.Mu(1,i), model.Sigma(1,1,i), [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) gaussPDF(0,0,model.Sigma(1,1,i))*model.Priors(i)], [.7 .7 .7], 1, 'h');
	Pt = Pt + Ptmp(2,:); 
end
axis([limAxes(1) limAxes(2) 0 gaussPDF(0,0,Sigma(1,1))]); %axis tight;
set(gca,'Xtick',[]); set(gca,'Ytick',[]);
xlabel('x_1');
ylabel('P(x_1)');
plotDistrib1D(Pt, [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) max(Pt)], [0 0 0], 1, 'h');


% %% Plot 2D
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% limAxes = [-.6 1.6 -.6 1.6];
% figure('PaperPosition',[0 0 4 4],'position',[10,10,1200,1200]); hold on; %axis off;
% plotGMM(model.Mu, model.Sigma,[0 0 0],.2);
% axis equal; axis(limAxes); 
% set(gca,'Xtick',[]); set(gca,'Ytick',[]);
% xlabel('x_1'); ylabel('x_2');
% 
% nbGrid = 200;
% [xx,yy] = meshgrid(linspace(limAxes(1),limAxes(2),nbGrid), linspace(limAxes(3),limAxes(4),nbGrid));
% z = zeros(nbGrid^2,1);
% for i=1:model.nbStates
%   z = z + model.Priors(i) * gaussPDF([xx(:)'; yy(:)'], model.Mu(:,i), model.Sigma(:,:,i));
% end
% zz = reshape(z,nbGrid,nbGrid);
% contour(xx,yy,zz,[.4,.4], 'color',[0 0 0],'linestyle','--','linewidth',1);
% 
% plotGMM(Mu, Sigma,[.8 0 0],.2);

%pause;
%close all;
