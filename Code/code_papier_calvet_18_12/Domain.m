termModel=['CANFCPv2']; hfun=['liborswap'];
hfunpar.dt=dt; hfunpar.ny=ny;
hfunpar.swapmat=swapmat; hfunpar.libormat=[];
nx=10;
hfunpar.nx=nx;
modelflag=[termModel,'_FS',num2str(nx)];
hfunpar.modelflag=modelflag;
par=[-2.5779516699490470e+00 -2.7279929650145163e-01 -2.8906286791881035e+00 -1.0241855036886078e+00 -1.3121658959363447e+00 -7.8233189124250764e+00 zeros(1,nx) ]';
fv = feval(termModel,par,hfunpar)
disp(fv)

disp(fv.Phi)
disp(fv.A)
disp(fv.Q)
