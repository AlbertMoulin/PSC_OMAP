%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nlnL, lnLv, predErrv,xEstv,yFitv] =     ratelikefunlf(par,data,hfun,filter, termModel,hfunpar)

%function [nlnL, lnLv, predErrv,xEstv,yFitv] = mainFunc_ekf(par,data,ffun, hfun,filter, termModel,varargin);
% Inputs:
%       par  --      parameters to be optimized
%       data --      observations (nobs by ny per day)
%       ffun --      state equation ffun(x, ffunpar)
%       hfun --      measurement equation hfun(x, t, hfunpar)
%       filter --    filtering technique: EKF, UKF, etc
%       termModel -- specific model to be estimated
%       varargin --- other parameters to be carried over to the termModel
% outputs:
%       nlnL  --     Negative of log likelihood function, to be minimized
%       lnLv  --     Log likehood function as a function of time; for standard error calculation
%       PredErrv-    Forecasting error on each observation
%       xEstv  ---   Extracted states
%       yFitv  ---   Fitted Observation
%  Liuren Wu; July 2002, revised April, 2009
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ffunpar, hfunpar,xEst,PEst, Q,R]=feval(termModel,par, hfunpar);

A=ffunpar.A;
F=ffunpar.Phi;

if isfinite(det(PEst))
    nx=length(xEst(:));
    [nobs, ny]=size(data);

    lnLv=zeros(nobs,1); predErrv = zeros(nobs, ny);   yFitv=predErrv; xEstv = zeros(nobs,nx);

    for t = 1:nobs
        t;
        y = data(t,:)';
        ind=find(isfinite(y)); R2=R(ind,ind); %pick out missing data
        [xEst,PEst,xPred,yPred,predErr,yVar] = feval(filter,y(ind),xEst,PEst,Q,R2,t,ind,A, F,hfun, hfunpar);
        dp=det(yVar);
        if isfinite(dp)
            % likelihood value
            lnLv(t) = -(log(dp) + predErr'*pinv(yVar)*predErr)/2;
            % outputs
            predErrv(t,ind) = predErr'; %predicting error
            xEstv(t,:) = xEst'; %state
            yFitv(t,ind) = feval(hfun,xEst,t,ind,hfunpar)';
        else
            lnLv=-1e10; disp('bad parameter');
            break; return;
        end

    end
    nlnL = -mean(lnLv(4:end));
    modelflag=hfunpar.modelflag;
    if exist(['C:\code\PSC_OMAP\Code\code_papier_calvet_18_12\output\nln_',modelflag,'.txt'],'file');
        nln0=load(['C:\code\PSC_OMAP\Code\code_papier_calvet_18_12\output\nln_',modelflag,'.txt']);
        if nlnL<nln0
            save(['C:\code\PSC_OMAP\Code\code_papier_calvet_18_12\output\nln',modelflag,'.txt'], 'nlnL', '-ascii','-double');
            save(['C:\code\PSC_OMAP\Code\code_papier_calvet_18_12\output\par_',modelflag,'.txt'], 'par', '-ascii','-double');
        end
    else
        save(['C:\code\PSC_OMAP\Code\code_papier_calvet_18_12\output\nln_',modelflag,'.txt'], 'nlnL', '-ascii','-double');
        save(['C:\code\PSC_OMAP\Code\code_papier_calvet_18_12\output\par_',modelflag,'.txt'], 'par', '-ascii','-double');

    end
else
    disp('bad parameter'); nlnL=1e10;
end
return