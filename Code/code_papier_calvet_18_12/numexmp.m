%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function function [H,C]=Cascade_coefficients(kappav,kappar, sigmar, nx, thetarp, thetar,mm)
% Compute coefficients on the cascade model
%  Author: Liuren Wu, liurenwu@gmail.com
%  Date: June 2009 and after
%  Reference: The Multifrequency scaling behavior of the interest rate term structure, working paper, Laurent Calvet, Adlai Fisher, Liuren Wu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
nx=10;
kappar=1/30; kappan=52; b=(kappan/kappar)^(1/(nx-1));
kappar=1/30; kappan=12; b=(kappan/kappar)^(1/(nx-1))

%% données résultant de l'optimisation du modèle, données de Calvet:
kappar=exp(-3); b=exp(exp(-5.5649999999999999e-01)); % paramètres issus de l'estimation

% ici, code matrice sous diagonale
Kappa=zeros(nx,nx);Kappa(nx,nx)=kappar;
kappav=zeros(nx,1); kappav(nx)=kappar;
for n=nx-1:-1:1
    kappav(n)=kappav(n+1)*b; % implémente les kappa de manière géométrique
    Kappa(n,n:n+1)=[kappav(n),-kappav(n)];
end

%code les response functions p.943 (chaque volonne est la future somme)
A=eye(nx); A(1,1)=1/kappav(1);
for j=1:nx-1
    for  i=1:j
        A(i,j+1)=-kappav(j)*A(i,j)/(kappav(i)-kappav(j+1));
    end
    A(j+1,j+1)=(kappav(j)/kappav(j+1))*(A(1:j,j)'*(kappav(1:j)./(kappav(1:j)-kappav(j+1))));
end

log(b)/(kappav(2)*(b-1))

mat=[1/52/12:1/360:200]; nm=length(mat);
Ki=repmat(kappav,1,nm).*exp(-kappav*mat);
a=Ki;
a(1,:)=A(1,1)*Ki(1,:);
for n=2:nx
    a(n,:)=sum(repmat(A(1:n,n),1,nm).*Ki(1:n,:));
end

%mat,exp(-kappav(3)*mat),
figure(1)
clf
%plot(log(mat),a)
semilogx(mat,a(end,:),'b-',mat,a(2,:),'k:',mat,a(1,:),'r--',mat,a(3:end-1,:),'k:','LineWidth',2)
xlabel('Time Horizon {\it\tau } (years)','FontSize',16)
ylabel('Response Function{\it a_j(\tau)}','FontSize',16)
legend('{\it j} = 1','{\it j} = 2-9','{\it j} = 10','Location','NorthEast')
legend boxoff
axis([1/52/2,100,0,1])
set(gca,'Box','on','LineWidth',2,'FontSize', 16)
print('-depsc', '-r70',['figlxffloading_numexmp.eps'])

figure(2)
clf
%plot(log(mat),a)
semilogx(mat,a(end,:),'b-',mat,a(2,:),'k:',mat,a(1,:),'r--',mat,a(3:end-1,:),'k:','LineWidth',2)
xlabel('Time Horizon {\it\tau } (years)','FontSize',16)
ylabel('Response Function, {\it a_j(\tau)}','FontSize',16)
legend('{\it j} = 1','{\it j} = 2-9','{\it j} = 10','Location','NorthEast')
legend boxoff
axis([1/52/2,100,0,1])
set(gca,'Box','on','LineWidth',2,'FontSize', 16)
print('-depsc', '-r70',['figlxffloading_numexmp2.eps'])

return