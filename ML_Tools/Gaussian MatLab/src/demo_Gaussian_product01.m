function demo_Gaussian_product01
% Product of Gaussians (standard and sequential computation)
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
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
nbVar = 2; %Number of variables
nbStates = 3; %Number of states


%% Set GMM parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% d(:,1) = [1; 2];
% d(:,2) = [3; -3];
% d(:,3) = [-2; 1];
% model.Mu = [0 2 0; 0 0 3];
% model.Sigma(:,:,1) = d(:,1)*d(:,1)' + eye(2)*1E-1;
% model.Sigma(:,:,2) = d(:,2)*d(:,2)' + eye(2)*1E-1;
% model.Sigma(:,:,3) = d(:,3)*d(:,3)' + eye(2)*1E-1;

d = randn(nbVar,nbStates);
model.Mu = randn(nbVar,nbStates);
for i=1:nbStates
	model.Sigma(:,:,i) = d(:,i)*d(:,i)' + eye(2)*1E-1;
end


%% Product of Gaussians (standard computation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SigmaTmp = zeros(nbVar);
MuTmp = zeros(nbVar,1);
for i=1:nbStates
	SigmaTmp = SigmaTmp + inv(model.Sigma(:,:,i));
	MuTmp = MuTmp + model.Sigma(:,:,i)\model.Mu(:,i);
end
SigmaP = inv(SigmaTmp);
MuP = SigmaP * MuTmp;


%% Product of Gaussians (sequential computation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SigmaTmp = zeros(nbVar);
MuTmp = zeros(nbVar,1);
for i=1:2
	SigmaTmp = SigmaTmp + inv(model.Sigma(:,:,i));
	MuTmp = MuTmp + model.Sigma(:,:,i)\model.Mu(:,i);
end
Sigma(:,:,1) = inv(SigmaTmp);
Mu(:,1) = Sigma(:,:,1) * MuTmp;
Sigma(:,:,2) = model.Sigma(:,:,3);
Mu(:,2) = model.Mu(:,3);

SigmaTmp = zeros(nbVar);
MuTmp = zeros(nbVar,1);
for i=1:2
	SigmaTmp = SigmaTmp + inv(Sigma(:,:,i));
	MuTmp = MuTmp + Sigma(:,:,i)\Mu(:,i);
end
SigmaP2 = inv(SigmaTmp);
MuP2 = SigmaP2 * MuTmp;


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,100,1200,600]); hold on; axis off;
plotGMM(model.Mu, model.Sigma, [0 .8 0]);
plotGMM(MuP, SigmaP, [.8 0 0]);
plotGMM(MuP2, SigmaP2, [0 0 .8]);
axis equal;

%pause;
%close all;

