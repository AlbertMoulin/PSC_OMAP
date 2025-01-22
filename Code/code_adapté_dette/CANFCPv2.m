%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ffunpar, hfunpar,xEst,PEst,Q,R] = CANFCPv2(par,hfunpar)
% 2c: the model is about right;scaling in Q, constant market price 
% 2cr: constrain kappanp>0
% 2cr3: scale in p, constant g0+g1*X
% 2cr4: scale in P, constant g0+g1(x2t-x1t); estimates are better
% 2cr6: g0 only.
% Use analytical formula
% Liuren Wu, liurenwu@gmail.com
% April, 2009 and after
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
swapmat=hfunpar.swapmat; %maturité des swaps
libormat=hfunpar.libormat;
dt=hfunpar.dt;
ny=hfunpar.ny;
nx=hfunpar.nx;

%paramètres de la p.947
npar=length(par(:)); % nombre de paramètres
par=reshape(par, npar,1); 
eyex=eye(nx); % nombre de facteurs = ici 10, mais peut changer en fonction du choix du modèle
epar=exp(par); 
kappar=epar(1); % 1 seul kappa, reste est une progression géom
sigmar=epar(2); % factor volatility. identique pour chaque facteur
thetarp=epar(3); % long run level of 1st factor
b=exp(epar(4)); % adjustment speed of geomtric progression of ks. Double exponentielle ?
gamma0=par(5); % market price of risks
R=epar(6)*eye(ny); % est renvoyé directement, pas utilisé sinon
gamma1=par(7:6+nx); % possiblement cas dans lequel on s'autorise à avoir des prix de facteurs de risque différents

gamma0v=gamma0*sigmar;
kappav=zeros(nx,1); kappav(nx)=kappar;
Kappa=zeros(nx,nx); Kappa(nx,nx)=kappar;
for n=nx-1:-1:1
    kappav(n)=kappav(n+1)*b; 
    Kappa(n,n:n+1)=[kappav(n),-kappav(n)];
end
Kappatheta=zeros(nx,1);Kappatheta(nx)=kappar*thetarp;
Kappas=Kappa-repmat(sigmar*gamma1(:)',nx,1);
Kappathetas=Kappatheta-gamma0v;
theta=Kappa\Kappatheta;
br=zeros(nx,1);br(1)=1;

SS2=sigmar^2*eye(nx);
Q=SS2*dt;
Phi = expm(-Kappa*dt); 
ffunpar.Phi=Phi;
A=(eyex-Phi)*theta;
ffunpar.A=A;
xEst = theta;
PEst = Q;

%%%%Measurement % implémentation de la p.944 --> calcul du prix
%%%%Measurement
h=2;hfunpar.h=h;
matv=[1/h:1/h:max(swapmat)]; nm=length(matv); at_swap=zeros(nm,1); bt_swap=zeros(nm,nx); 
[U,D]=eig(Kappas); invU=inv(U);d=diag(D);% U = vecteurs propres, D = valeurs propres
epd=exp(d); lepdv=log(epd*epd');vvs=invU*invU'; 
invKappas=inv(Kappas); invKappaspbr=invKappas'*br;
for k=1:nm
    t=matv(k);
    epd=exp(-d*t); epdv=epd*epd'; 
    VVs=U*(vvs.*(1-epdv)./lepdv)*U';
    IKappas=eyex-expm(-Kappas*t); 
    IKappasp=eyex-expm(-Kappas'*t); 
    
    btv=IKappasp*invKappaspbr;
    atv=  Kappathetas'*invKappaspbr*t -Kappathetas'*invKappas'*IKappasp*invKappaspbr   ...
        -0.5*br'*invKappas*SS2*invKappas'*br*t  ...
        +0.5*br'*invKappas*SS2*invKappas'*IKappasp*invKappaspbr ...
        +0.5*br'*invKappas*SS2*invKappas*IKappas*invKappaspbr ...
        -0.5*br'*invKappas*SS2*VVs*invKappaspbr;
    bt_swap(k,:)=real(btv');    at_swap(k,1)=real(atv);
end

hfunpar.at_swap=at_swap;
hfunpar.bt_swap=bt_swap;
hfunpar.swapmatv=matv';

nlibor=length(libormat);

at_libor=zeros(nlibor,1); bt_libor=zeros(nlibor,nx); 
for k=1:nlibor
    t=libormat(k);
    epd=exp(-d*t); epdv=epd*epd';
    VVs=U*(vvs.*(1-epdv)./lepdv)*U';
    IKappas=eyex-expm(-Kappas*t);
    IKappasp=eyex-expm(-Kappas'*t);

    btv=IKappasp*invKappaspbr;
    atv=  Kappathetas'*invKappaspbr*t -Kappathetas'*invKappas'*IKappasp*invKappaspbr   ...
        -0.5*br'*invKappas*SS2*invKappas'*br*t  ...
        +0.5*br'*invKappas*SS2*invKappas'*IKappasp*invKappaspbr ...
        +0.5*br'*invKappas*SS2*invKappas*IKappas*invKappaspbr ...
        -0.5*br'*invKappas*SS2*VVs*invKappaspbr;
    bt_libor(k,:)=real(btv');    at_libor(k,1)=real(atv);
end

hfunpar.at_libor=at_libor;
hfunpar.bt_libor=bt_libor;