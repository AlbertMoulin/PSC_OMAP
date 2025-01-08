%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  estimation02c
%  the right one: Scale in kappa_Q, constant sigma,  add the same market price g0+g1X to all risk sources.
% Liuren Wu, liurenwu@gmail.com
% April, 2009 and after
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;format compact;format short;

estimation=1; unc=1; % unconstrained optimization
stderror=1;
gammavplot=1;

tol=1e-4; nit=50000;
fopt=optimset('Display','iter','MaxIter',nit,'MaxFunEvals',nit,'TolX', tol, 'TolFun', tol);

filter='ukf_lfnlh'; likefun='ratelikefunlf';

%load the data
load(['../data_dette/nusrates_dette.mat'],'rates','mat','mdate','-mat');


cdate=[mdate(1):mdate(end)]';
wdate=cdate(weekday(cdate)==4);dt=1/52;



disp( size( rates(:,:) )  )
disp(size(mdate))

%load the data
load(['../data/nusrates.mat'],'rates','mat','mdate','-mat');


cdate=[mdate(1):mdate(end)]';
wdate=cdate(weekday(cdate)==4);dt=1/52;

%disp(wdate)
%disp(rates(1:10,:))

