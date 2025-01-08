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



rates=interp1(mdate,rates(:,:));
libormat=6;
%mat=[6/12;swapmat];
[T,ny]=size(rates)
datevec([wdate(1);wdate(end)])
lastdate=datestr(wdate(end),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%(1) Three-factor Gauss-Affine Model
termModel=['CANFCPv2']; hfun=['liborswap'];
hfunpar.dt=dt; hfunpar.ny=ny;
hfunpar.swapmat=swapmat; hfunpar.libormat=libormat'/12;
nx=10;
hfunpar.nx=nx;
modelflag=[termModel,'_FS',num2str(nx)];
hfunpar.modelflag=modelflag;
if exist(['../output/par_',modelflag,'.txt'],'file');
    par=max(-10,min(10,load(['../output/par_',modelflag,'.txt'])));
else
    par=[-3.2484   -4.1377   -3.8077   -0.4693  -0.2820 -9.6393 zeros(1,nx) ]';
    par=[  -2.9702   -4.2022   -9.9750   -0.5565   -0.1706   -9.6040   -0.2710    0.1458    0.0338    0.0892    3.7794   -0.0607   -0.2011   -0.9491   -1.6781    0.0099]';
end
epar=exp(par);
kappar=epar(1);
sigmar=epar(2);
thetarp=epar(3);
b=exp(epar(4));
gamma0=par(5);
R=epar(6)*eye(ny);
gamma1=par(7:6+nx);
gamma0v=gamma0*sigmar;

    
t0=clock;
[loglike,likeliv, predErr,mu_dd,y_dd]=feval(likefun, par,rates,hfun,filter,termModel,hfunpar);loglike
runtime=etime(clock,t0)
if estimation
    if unc
        par=fminunc(likefun,par,fopt,rates, hfun,filter,termModel,hfunpar);
    else
        par=fminsearch(likefun,par,fopt,rates,hfun,filter,termModel,hfunpar);
    end
    [loglike,likeliv, predErr,mu_dd,y_dd]=feval(likefun, par,rates, hfun,filter,termModel,hfunpar);
    save(['../output/par_',modelflag,'.txt'], 'par', '-ascii','-double');
    [loglike,likeliv, predErr,mu_dd,y_dd]=feval(likefun, par,rates,hfun,filter,termModel,hfunpar);loglike
    save(['../output/nln_',modelflag,'.txt'], 'loglike', '-ascii','-double');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if stderror
    npar=length(par);
    h=zeros(T, npar);
    stepsizei=1e-4; %sqrt(eps);
    for i=1:npar
        bb1=par; bb2=par;
        bb1(i)=par(i)-stepsizei/2;
        bb2(i)=par(i)+stepsizei/2;
        [temp, x1]=feval(likefun, bb1,rates, hfun,filter,termModel,hfunpar);
        [temp, x2]=feval(likefun, bb2,rates, hfun,filter,termModel,hfunpar);
        h(:,i)=(x2-x1)./(stepsizei);
    end
    [Q,R]= qr(h,0);Rinv = pinv(R);
    %Rinv=pinv(h'*h);
    avar = Rinv*Rinv';
    stdpar=sqrt(diag(avar));
    
%     JOP=(h'*h)/T;
%     vcov=pinv(JOP)/T;
%     stdpar=sqrt(diag(vcov));
    
    
    epar=exp(par);
    kappar=epar(1);
    sigmar=epar(2);
    thetarp=epar(3);
    b=exp(epar(4));
    gamma0=par(5);
    R=epar(6);
    gamma1=par(7:6+nx);
    
    parpr=exp(par); stdpp=parpr.*stdpar;
    parpr(4)=b; stdpp(4)=b*epar(4)*stdpar(4);
    ind=[5,7:6+nx];
    parpr(ind)=par(ind);stdpp(ind)=stdpar(ind);
    table= [parpr stdpp];
    ind=[1:5,7:6+nx]';
    fprintf(1,'   \n');
    fprintf(1,'   &  %7.4f  &  ( %7.4f )   \\\\ \n', table(ind,:)');
    
end
if gammavplot
    epar=exp(par);
    kappar=epar(1);
    sigmar=epar(2);
    thetarp=epar(3);
    b=exp(epar(4));
    gamma0=par(5);
    R=epar(6);
    gamma1=par(7:6+nx);
    kappav=zeros(nx,1); kappav(nx)=kappar;
    for n=nx-1:-1:1
        kappav(n)=kappav(n+1)*b;
    end
    fv=[nx:-1:1];
    
    figure(1)
    clf
    plot(fv,gamma1,'o-', 'LineWidth',2,'MarkerSize',10)
    xlabel('Frequency, j','FontSize',16)
    ylabel('\lambda_j','FontSize',16)
    grid
    set(gca,'Box','on','LineWidth',2,'FontSize', 16)
    print('-depsc','-r70', ['../JFQAR1/figgammav_',modelflag,'.eps'])
    
    lk=-log(kappav);
    figure(2)
    plot(lk,gamma1,'o-', 'LineWidth',2,'MarkerSize',10);grid
    ylabel('\lambda_j','FontSize',16)
    xlabel('ln \kappa_j','FontSize',16)
    set(gca,'Box','on','LineWidth',2,'FontSize', 16)
    print('-depsc','-r70', ['../JFQAR1/figlnkappavgammav_',modelflag,'.eps'])
    
end
return
