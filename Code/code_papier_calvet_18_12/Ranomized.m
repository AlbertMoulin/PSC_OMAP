% filepath: /d:/Albert/Polytechnique/PSC/Code/code_papier_calvet_18_12/Randomized.m
start = 283; % Start index of the parameters
% Monte Carlo simulation around the initial point par
initial_par = [ -2.4213571986670770e+00
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

par = zeros(16, 1);

num_simulations = 100; % Number of Monte Carlo simulations
perturbation_scale = 5; % Scale of the perturbation

loglike_values = zeros(num_simulations, 1);

for i = start : start + num_simulations
    % Perturb the initial parameters
    perturbed_par = initial_par + perturbation_scale * randn(size(initial_par));
    % Evaluate the log-likelihood function
    loglike_values(i) = estimationCANFCPv2SFunction(perturbed_par, 1, 1, 0, 0, 0, 0, num2str(100+i));
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



