par = [ -2.4213571986670770e+00
        -2.6485877467981833e-01
        -9.8066105033629281e-01
        -1.0889928095144858e+00
        -1.2696723247784476e+00
        -7.8321557692519495e+00
        3.2775357009488045e-03
        4.2029577807859951e-03
        8.8513044713688052e-02
        -3.8868574923438301e-02
        -3.3190659949449849e-02
        1.0202104711546502e-02
        -4.8954539432252836e-02
        -1.3474447135741901e-02
        -7.1531751170157329e-04
        -2.1388395992296989e-02];

estimation = 1
unc = 1
stderror = 0
gammavplot = 0
draw = 0
prediction = 0
AttemptNumber = 'A1'


tol=1e-4; nit=50000;
fopt=optimset('Display','iter','MaxIter',nit,'MaxFunEvals',nit,'TolX', tol, 'TolFun', tol);

% Utiliser une fonction anonyme pour passer les arguments supplémentaires
objectiveFunction = @(p) estimationCANFCPv2SFunction(p, estimation, unc, stderror, gammavplot, draw, prediction, AttemptNumber);

% Appeler fminunc avec la fonction anonyme
par = fminunc(objectiveFunction, par, fopt);