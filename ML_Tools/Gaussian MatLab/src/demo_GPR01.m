function demo_GPR01
% Gaussian process regression (GPR) 
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVarX = 1; %Dimension of x
nbVar = nbVarX+1; %Dimension of datapoint (x,y)
nbData = 4; %Number of datapoints
nbDataRepro = 100; %Number of datapoints in a reproduction
nbRepros = 20; %Number of randomly sampled reproductions
p(1)=1E0; p(2)=1E-1; p(3)=1E-3; %GPR parameters (here, for squared exponential kernels and noisy observations)


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = linspace(0,1,nbData);
y = randn(1,nbData) * 1E-1;
Data = [x; y];


%% Reproduction with GPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GPR precomputation (here, with a naive implementation of the inverse)
K = covFct(x, x, p);
invK = pinv(K + p(3) * eye(size(K))); %Inclusion of noise on the inputs

%Mean trajectory computation
xs = linspace(0,1,nbDataRepro);
Ks = covFct(xs, x, p);
r(1).Data = [xs; (Ks * invK * y')']; 

%Uncertainty evaluation
Kss = covFct(xs, xs, p);
S = Kss - Ks * invK * Ks';
r(1).SigmaOut = zeros(nbVar-1,nbVar-1,nbData);
for t=1:nbDataRepro
	r(1).SigmaOut(:,:,t) = eye(nbVarX) * S(t,t); 
end

%Generate stochastic samples from the prior 
[V,D] = eig(Kss);
for n=2:nbRepros/2
	yp = real(V*D^.5) * randn(nbDataRepro,1)*2E-1; 
	r(n).Data = [xs; yp'];
end

%Generate stochastic samples from the posterior 
[V,D] = eig(S);
for n=nbRepros/2+1:nbRepros
	ys = real(V*D^.5) * randn(nbDataRepro,1)*0.5 + r(1).Data(2,:)'; 
	r(n).Data = [xs; ys'];
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 12 4],'position',[10 10 1300 600]); 
limAxes = [0, 1, -.5 .5];

%Prior samples
subplot(1,3,1); hold on; title('Samples from prior','fontsize',14);
for n=2:nbRepros/2
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','lineWidth',3.5,'color',[.9 .9 .9]*rand(1));
end
set(gca,'xtick',[],'ytick',[]); axis(limAxes);
xlabel('x_1'); ylabel('y_1');

%Posterior samples
subplot(1,3,2); hold on;  title('Samples from posterior','fontsize',14);
for n=nbRepros/2+1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','lineWidth',3.5,'color',[.9 .9 .9]*rand(1));
end
plot(Data(1,:), Data(2,:), '.','markersize',24,'color',[1 0 0]);
set(gca,'xtick',[],'ytick',[]); axis(limAxes);
xlabel('x_1'); ylabel('y_1');

%Trajectory distribution
subplot(1,3,3); hold on;  title('Trajectory distribution','fontsize',14);
patch([r(1).Data(1,:), r(1).Data(1,end:-1:1)], ...
	[r(1).Data(2,:)+squeeze(r(1).SigmaOut.^.5)', r(1).Data(2,end:-1:1)-squeeze(r(1).SigmaOut(:,:,end:-1:1).^.5)'], ...
	[.8 .8 .8],'edgecolor','none');
plot(r(1).Data(1,:), r(1).Data(2,:), '-','lineWidth',3.5,'color',[0 0 0]);
plot(Data(1,:), Data(2,:), '.','markersize',24,'color',[1 0 0]);
set(gca,'xtick',[],'ytick',[]); axis(limAxes);
xlabel('x_1'); ylabel('y_1');

%pause;
%close all;
end

% %User-defined distance function
% function d = distfun(x,y)
% 	d = min(x,y) + .1;
% end

function K = covFct(x, y, p)
% 	K = pdist2(x', y', @distfun); %User-defined distance function
	K = p(1) .* exp(-p(2)^-1 .* pdist2(x',y').^2);
end



