%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  estimation02c
%  the right one: Scale in kappa_Q, constant sigma,  add the same market price g0+g1X to all risk sources.
% Liuren Wu, liurenwu@gmail.com
% April, 2009 and after
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;format compact;format short;

estimation=0; unc=0; % unconstrained optimization
stderror=1; % sert à quoi ?
dataDette = 0;
gammavplot=0;
draw =1;
prediction=0;



tol=1e-4; nit=50000;
fopt=optimset('Display','iter','MaxIter',nit,'MaxFunEvals',nit,'TolX', tol, 'TolFun', tol);

filter='ukf_lfnlh'; likefun='ratelikefunlf';

%load the data

if dataDette
    load('code_papier_calvet_18_12\data_dette_cleaned\nusrates_dette_cleaned.mat','rates','swapmat','mdate','-mat');
    cdate=[mdate(end):mdate(1)]';
    wdate=cdate(weekday(cdate)==4);dt=1/52;
    rates=interp1(mdate,rates,wdate);
    mat=[swapmat];
    [T,ny]=size(rates)
    datevec([wdate(1);wdate(end)])
    lastdate=datestr(wdate(end),1);
    %(1) Three-factor Gauss-Affine Model
    termModel=['CANFCPv2']; hfun=['liborswap'];
    hfunpar.dt=dt; hfunpar.ny=ny;
    hfunpar.swapmat=swapmat; hfunpar.libormat=[];
    nx=10;
    hfunpar.nx=nx;
    modelflag=[termModel,'_FS',num2str(nx)];
    hfunpar.modelflag=modelflag;
    if exist(['code_papier_calvet_18_12\output\par_',modelflag,'.txt'],'file');
        par= load(['code_papier_calvet_18_12\output\par_',modelflag,'.txt']);
    else
        %par=[-3.0   -4.0   -3.0   -0.2  -0.1 -9.0 zeros(1,nx) ]';
        % par = [  -2.1484810243465255e+00
        % -3.1513019673402223e+00
        % -3.7827252671688236e+00
        % -7.7655162827575608e-01
        % -6.5077213734018691e-01
        % -7.7870583588909650e+00
        % -8.9128061610756737e-01
        % -7.6038747082274305e-01
        % -9.4590691999576437e-01
        % -1.3339308199433615e+00
        % -1.3261754231503742e+00
        % -1.3445743165930382e+00
        % -1.7148042866277728e+00
        % -7.7747539375937103e-01
        %  6.1790848796928743e-01
        %  2.7486204588319798e+00]';
        par=[-2.15   -3.15   -3.78   -0.78  -0.65 -7.79 zeros(1,nx) ]';
    end
else
    load('code_papier_calvet_18_12\data\nusrates.mat','rates','mat','swapmat','libormat','mdate','-mat');
    cdate=[mdate(1):mdate(end)]';
    wdate=cdate(weekday(cdate)==4);dt=1/52;
    rates=interp1(mdate,rates(:,[4,7:end]),wdate);
    libormat=6;
    mat=[6/12;swapmat];
    [T,ny]=size(rates)
    datevec([wdate(1);wdate(end)])
    lastdate=datestr(wdate(end),1);
    %(1) Three-factor Gauss-Affine Model
    termModel=['CANFCPv2']; hfun=['liborswap'];
    hfunpar.dt=dt; hfunpar.ny=ny;
    hfunpar.swapmat=swapmat; hfunpar.libormat=libormat'/12;
    nx=10;
    hfunpar.nx=nx;
    modelflag=[termModel,'_FS',num2str(nx)];
    hfunpar.modelflag=modelflag;
    if 1==0;
        par=max(-10,min(10,load(['code_papier_calvet_18_12\output\par_',modelflag,'.txt'])));
    else
        %par=[-3.0   -4.0   -3.0   -0.2  -0.1 -9.0 zeros(1,nx) ]';
        par=[  -2.9702   -4.2022   -9.9750   -0.5565   -0.1706   -9.6040   -0.2710    0.1458    0.0338    0.0892    3.7794   -0.0607   -0.2011   -0.9491   -1.6781    0.0099]';
        %par=[  -3   -4.2022   -9.9750   -0.5565   -0.1706   -9.6040   -0.2710    0.1458    0.0338    0.0892    3.7794   -0.0607   -0.2011   -0.9491   -1.6781    0.0099]';
    
    end
    par=[  -2.9702   -4.2022   -9.9750   -0.5565   -0.1706   -9.6040   -0.2710    0.1458    0.0338    0.0892    3.7794   -0.0607   -0.2011   -0.9491   -1.6781    0.0099]';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epar=exp(par);
kappar=epar(1);
sigmar=epar(2);
thetarp=epar(3);
b=exp(epar(4));
gamma0=par(5);
R=epar(6)*eye(ny);
gamma1=par(7:6+nx);
gamma0v=gamma0*sigmar;


%% Ici, évalue la vraissemblance pour un filtre et un modèle donné.
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
    save(['code_papier_calvet_18_12\output\par_',modelflag,'.txt'], 'par', '-ascii','-double');
    [loglike,likeliv, predErr,mu_dd,y_dd]=feval(likefun, par,rates,hfun,filter,termModel,hfunpar);loglike %redondant ? 
    save(['code_papier_calvet_18_12\output\mu_dd_',modelflag,'.txt'], 'mu_dd', '-ascii','-double');
    save(['code_papier_calvet_18_12\output\y_dd_',modelflag,'.txt'], 'y_dd', '-ascii','-double');

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

    % for unknown reasons the parameters are given as a line with our data and not as a column
    if size(par,1)==1
        parpr=exp(par)'; 
    else
        parpr=exp(par); 
    end
    
    stdpp=parpr.*stdpar;
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
    print('-depsc','-r70', ['code_papier_calvet_18_12\JFQAR1\figgammav_',modelflag,'.eps'])
    
    lk=-log(kappav);
    figure(2)
    plot(lk,gamma1,'o-', 'LineWidth',2,'MarkerSize',10);grid
    ylabel('\lambda_j','FontSize',16)
    xlabel('ln \kappa_j','FontSize',16)
    set(gca,'Box','on','LineWidth',2,'FontSize', 16)
    print('-depsc','-r70', ['code_papier_calvet_18_12\JFQAR1\figlnkappavgammav_',modelflag,'.eps'])
    
end

drawperiod=length(wdate)-100:length(wdate);
if draw %draws the yield curve
    [loglike,likeliv, predErr,mu_dd,y_dd]=feval(likefun, par,rates,hfun,filter,termModel,hfunpar);
    figure(3)
    clf
    maturities = mat; % Combine libormat and swapmat for maturities
    if dataDette
        columns = [1, 6, 10];
    else
        columns = [1, 3, 10];
    end

    colors = {[0 0 0.5], [0 0.5 0], [0.5 0 0]}; % Dark blue, dark green, black
    for i = 1:length(columns)
        subplot(3, 1, i)
        plot(wdate(drawperiod), rates(drawperiod), 'LineWidth',1, 'Color', colors{i})
        hold on
        plot(wdate(drawperiod), y_dd(drawperiod), 'r--', 'LineWidth',1)
        hold off
        datetick('x','mmmyy')
        grid
        legendLabels = sprintf('Maturity %.1f years', maturities(columns(i)));
        legend({legendLabels, ['Model ' legendLabels]}, 'Location', 'Best')
        ylabel('Yield (%)', 'FontSize', 16) % Indicate that the y-axis is in percentage
        set(gca,'Box','on','LineWidth',2,'FontSize', 16)
    end
    xlabel('Date', 'FontSize', 16) % Use the same x-axis for all subplots
    print('-depsc','-r70', ['code_papier_calvet_18_12\JFQAR1\figyieldcurve_',modelflag,'.eps'])
    figure(4)
    clf
    for i = 1:length(columns)
        subplot(3, 1, i)
        plot(wdate(drawperiod), rates(drawperiod,columns(i))-y_dd(drawperiod,columns(i)), 'LineWidth',2)
        datetick('x','mmmyy')
        grid
        legendLabels = sprintf('Error in Maturity %.1f years', maturities(columns(i)));
        legend(legendLabels, 'Location', 'Best')
        set(gca,'Box','on','LineWidth',2,'FontSize', 16)
    end
    print('-depsc','-r70', ['code_papier_calvet_18_12\JFQAR1\figyieldcurveerror_',modelflag,'.eps'])
end

T=10;



if prediction

    [ffunpar, hfunpar,xEst,PEst, Q,R]=feval(termModel,par, hfunpar); %nécessaire pour récup ffunpar


    PredY = zeros(T, ny);
    PredX = zeros(T, nx);
    X=mu_dd(end,:)';

    for i = 1:2
        y = rates(5,:)';
        test=find(isfinite(y));
        A=ffunpar.A;
        phi=ffunpar.Phi;
        Q=ffunpar.Q;
        X = A + phi*X + sqrt(Q)*randn(nx,1); %state propagation

        y_suivant = liborswap(X, [1,2],test, hfunpar); %measurement prediction
        PredY(i,:) = y_suivant;	
        PredX(i,:) = X;
        ind=find(isfinite(y_suivant));
% Plot the fitted yield over the last 10 weeks
figure(5)
clf
subplot(2, 1, 1)
plot(wdate(end-9:end), rates(end-9:end, 1), 'LineWidth', 2, 'DisplayName', 'Fitted Yield')
hold on
plot(wdate(end-9:end), y_dd(end-9:end, 1), 'r--', 'LineWidth', 2, 'DisplayName', 'Model Fitted Yield')
hold off
datetick('x', 'mmmyy')
grid
legend('show', 'Location', 'Best')
ylabel('Yield (%)', 'FontSize', 16)
title('Fitted Yield over the Last 10 Weeks', 'FontSize', 16)
set(gca, 'Box', 'on', 'LineWidth', 2, 'FontSize', 16)

% Plot the predicted yield for the next 10 weeks
subplot(2, 1, 2)
% Plot the fitted yield over the last 10 weeks
figure(5)
clf
subplot(2, 1, 1)
plot(wdate(end-9:end), rates(end-9:end, 1), 'LineWidth', 2, 'DisplayName', 'Fitted Yield')
datetick('x', 'mmmyy')
grid
legend('show', 'Location', 'Best')
ylabel('Yield (%)', 'FontSize', 16)
title('Fitted Yield over the Last 10 Weeks', 'FontSize', 16)
set(gca, 'Box', 'on', 'LineWidth', 2, 'FontSize', 16)

% Plot the predicted yield for the next 10 weeks
subplot(2, 1, 2)
future_dates = wdate(end) + (1:T)' * 7; % Calculate future dates (weekly intervals)
plot(future_dates, PredY(:, 5), 'LineWidth', 2, 'DisplayName', 'Predicted Yield')
datetick('x', 'mmmyy')
grid
legend('show', 'Location', 'Best')
ylabel('Yield (%)', 'FontSize', 16)
xlabel('Weeks Ahead', 'FontSize', 16)
title('Predicted Yield for the Next 10 Weeks', 'FontSize', 16)
set(gca, 'Box', 'on', 'LineWidth', 2, 'FontSize', 16)
    end
end



return