% Define the domains for each component
domains = [
    0, 1;    % Domain for the 1st component
    -5, 5;   % Domain for the 2nd component
    10, 20;  % Domain for the 3rd component
    -10, 0;  % Domain for the 4th component
    100, 200;% Domain for the 5th component
    0.5, 1.5 % Domain for the 6th component
];

% Generate the random vector
randomVector = arrayfun(@(i) domains(i,1) + (domains(i,2) - domains(i,1)) * rand, 1:size(domains, 1));

% Display the random vector
disp(randomVector);