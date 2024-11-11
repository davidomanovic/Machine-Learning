function demo_LWR01
% Polynomial fitting with locally weighted regression (LWR). 
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
model.nbStates = 4; %Number of activation functions (i.e., number of states in the GMM)
model.nbVarIn = 2; %Degree of the polynomial (based on time input)
model.nbVarOut = 2; %Number of motion variables [x1,x2] 
nbData = 200; %Length of a trajectory
nbSamples = 5; %Number of demonstrations
tIn = linspace(0,1,nbData); %Input data for LWR


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data s(n).Data]; %Concatenation of the multiple demonstrations
end


%% Setting of the basis functions and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Set centroids equally spread in time
model = init_GMM_timeBased(tIn, model);

%Set constant shared covariance
for i=1:model.nbStates
	model.Sigma(:,:,i) = 1E-2; 
end

%Compute activation weights
H = zeros(model.nbStates,nbData);
for i=1:model.nbStates
	H(i,:) = gaussPDF(tIn, model.Mu(:,i), model.Sigma(:,:,i));
end
H = H ./ repmat(sum(H,1),model.nbStates,1);
Hn = repmat(H,1,nbSamples);

%Locally weighted regression 
Y = Data';
X = [];
Xr = [];
%Transformation of input data into polynomial basis functions
for d=0:model.nbVarIn
	X = [X, repmat(tIn.^d,1,nbSamples)'];
	Xr = [Xr, tIn.^d']; 
end
%Weighted least squares
for i=1:model.nbStates
	W = diag(Hn(i,:));
	MuP(:,:,i) = X'*W*X \ X'*W * Y; 
end
%Reconstruction of signal
Yr = zeros(nbData,model.nbVarOut);
for t=1:nbData
	for i=1:model.nbStates
		Yr(t,:) = Yr(t,:) + H(i,t) * Xr(t,:) * MuP(:,:,i);
	end
end
r(1).Data = Yr';


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 16 4],'position',[10,10,1300,500],'color',[1 1 1]); 
xx = round(linspace(1,64,model.nbStates));
clrmap = colormap('jet')*0.5;
clrmap = min(clrmap(xx,:),.9);

%Spatial plot
axes('Position',[0 0 .2 1]); hold on; axis off;
plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.7 .7 .7]);
plot(r(1).Data(1,:),r(1).Data(2,:),'.','markersize',16,'linewidth',3,'color',[.8 0 0]);
axis square; axis equal;

%Timeline plot 
axes('Position',[.25 .58 .7 .4]); hold on; 
for n=1:nbSamples
	plot(tIn, Data(1,(n-1)*nbData+1:n*nbData), '-','linewidth',1,'color',[.7 .7 .7]);
end
[~,id] = max(H,[],1);
for i=1:model.nbStates
	Xr = [];
	for d=0:model.nbVarIn
		Xr = [Xr, tIn(id==i).^d']; 
	end
	plot(tIn(id==i), Xr*MuP(:,1,i), '.','linewidth',6,'markersize',26,'color',min(clrmap(i,:)+0.5,1));
end
plot(tIn, Yr(:,1), '-','linewidth',2,'color',[.8 0 0]);
axis([min(tIn) max(tIn) min(Data(1,:))-1 max(Data(1,:))+1]);
ylabel('y_{t,1}','fontsize',16);

%Timeline plot of the basis functions activation
axes('Position',[.25 .12 .7 .4]); hold on; 
for i=1:model.nbStates
	patch([tIn(1), tIn, tIn(end)], [0, H(i,:), 0], min(clrmap(i,:)+0.5,1), 'EdgeColor', min(clrmap(i,:)+0.2,1), 'linewidth',2);
end
axis([min(tIn) max(tIn) 0 1.1]);
xlabel('t','fontsize',16); 
ylabel('\phi(x_t)','fontsize',16);

%print('-dpng','graphs/demo_LWR01.png');
%pause;
%close all;
