% filepath: /d:/Albert/Polytechnique/PSC/Code/code_papier_calvet_18_12/Randomized.m
start = 24; % Start index of the parameters
% Monte Carlo simulation around the initial point par
initial_par = [ -3.5843988315086515e+00
-4.3870502219328147e+00
-3.2210840921211790e+00
-7.3905285844054935e-01
-9.1219199537896400e-01
-7.7040657768169210e+00
-1.8224572669243677e-01
 1.5678935979216826e+00
 1.0948829150274473e+00
-2.3863116622385361e+00
 1.0467617883385074e+00
-1.6967985387130391e+00
 2.0832612020672157e+00
-4.9506267759207709e+00
-4.6989045759662833e+00
-6.5785088884506271e+00];

par = zeros(16, 1);

num_simulations = 100; % Number of Monte Carlo simulations
perturbation_scale = 2; % Scale of the perturbation

loglike_values = zeros(num_simulations, 1);

for i = start : start + num_simulations
    % Perturb the initial parameters
    perturbed_par = par;
    perturbed_par(1:6) = initial_par(1:6) + perturbation_scale * randn(6, 1);
    % Evaluate the log-likelihood function
    loglike_values(i) = estimationCANFCPv2SFunction(perturbed_par, 1, 1, 0, 0, 0, 0, num2str(i));
end

% Display the results
mean_loglike = mean(loglike_values);
std_loglike = std(loglike_values);

fprintf('Mean log-likelihood: %f\n', mean_loglike);
fprintf('Standard deviation of log-likelihood: %f\n', std_loglike);

% Plot the distribution of log-likelihood values
figure;
histogram(loglike_values, 30);
xlabel('Log-Likelihood');
ylabel('Frequency');
title('Distribution of Log-Likelihood Values from Monte Carlo Simulation');
grid on;



