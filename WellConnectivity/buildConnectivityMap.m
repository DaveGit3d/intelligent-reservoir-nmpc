function connMap = buildConnectivityMap()
% BUILD_CONNECTIVITY_MAP: Create connectivity map from Norne field data
%
% Based on your connectivity diagram showing:
%   - I1: Highly connected (P1, P2, P3)
%   - I2: Moderate (P1, P2)
%   - I3: Moderate (P2, P3)
%   - I4: Moderate (P3, P4)
%   - I5: Strong (P4, P5)
%   - I6: Weak (P4, P5)
%
% OUTPUT:
%   connMap : Struct with connectivity data
%             connMap.P1.inj{i}.name = 'I1'
%             connMap.P1.inj{i}.strength = 0.85 (0-1 scale)
%
% USAGE:
%   connMap = buildConnectivityMap();
%   weights_P1 = getInjectorWeights(connMap, 'P1');

%% ========== INITIALIZE STRUCTURE ==========
connMap = struct();

%% ========== PRODUCER P1 ==========
% From connectivity diagram: I1 (strong), I2 (moderate)
connMap.P1.inj{1}.name = 'I1';
connMap.P1.inj{1}.strength = 0.95;  % Very strong connection
connMap.P1.inj{1}.distance_m = 450;  % Approximate distance

connMap.P1.inj{2}.name = 'I2';
connMap.P1.inj{2}.strength = 0.65;  % Moderate connection
connMap.P1.inj{2}.distance_m = 750;

% Weak influence from I3
connMap.P1.inj{3}.name = 'I3';
connMap.P1.inj{3}.strength = 0.25;
connMap.P1.inj{3}.distance_m = 1200;

% Very weak influence from I4
connMap.P1.inj{4}.name = 'I4';
connMap.P1.inj{4}.strength = 0.15;
connMap.P1.inj{4}.distance_m = 1500;

%% ========== PRODUCER P2 ==========
% From diagram: I1 (strong), I2 (moderate), I3 (strong)
connMap.P2.inj{1}.name = 'I1';
connMap.P2.inj{1}.strength = 0.85;
connMap.P2.inj{1}.distance_m = 600;

connMap.P2.inj{2}.name = 'I2';
connMap.P2.inj{2}.strength = 0.70;
connMap.P2.inj{2}.distance_m = 500;

connMap.P2.inj{3}.name = 'I3';
connMap.P2.inj{3}.strength = 0.90;  % Strong connection
connMap.P2.inj{3}.distance_m = 400;

connMap.P2.inj{4}.name = 'I4';
connMap.P2.inj{4}.strength = 0.40;
connMap.P2.inj{4}.distance_m = 900;

%% ========== PRODUCER P3 ==========
% From diagram: I1 (moderate), I3 (moderate), I4 (strong)
connMap.P3.inj{1}.name = 'I1';
connMap.P3.inj{1}.strength = 0.60;
connMap.P3.inj{1}.distance_m = 800;

connMap.P3.inj{2}.name = 'I3';
connMap.P3.inj{2}.strength = 0.75;
connMap.P3.inj{2}.distance_m = 550;

connMap.P3.inj{3}.name = 'I4';
connMap.P3.inj{3}.strength = 0.85;  % Strong connection
connMap.P3.inj{3}.distance_m = 450;

connMap.P3.inj{4}.name = 'I5';
connMap.P3.inj{4}.strength = 0.30;
connMap.P3.inj{4}.distance_m = 1100;

%% ========== PRODUCER P4 ==========
% From diagram: I4 (moderate), I5 (strong), I6 (weak)
connMap.P4.inj{1}.name = 'I4';
connMap.P4.inj{1}.strength = 0.55;
connMap.P4.inj{1}.distance_m = 700;

connMap.P4.inj{2}.name = 'I5';
connMap.P4.inj{2}.strength = 0.95;  % Very strong connection
connMap.P4.inj{2}.distance_m = 350;

connMap.P4.inj{3}.name = 'I6';
connMap.P4.inj{3}.strength = 0.40;  % Weak connection
connMap.P4.inj{3}.distance_m = 950;

%% ========== PRODUCER P5 ==========
% From diagram: I5 (moderate), I6 (moderate)
connMap.P5.inj{1}.name = 'I5';
connMap.P5.inj{1}.strength = 0.70;
connMap.P5.inj{1}.distance_m = 600;

connMap.P5.inj{2}.name = 'I6';
connMap.P5.inj{2}.strength = 0.65;
connMap.P5.inj{2}.distance_m = 650;

connMap.P5.inj{3}.name = 'I4';
connMap.P5.inj{3}.strength = 0.25;  % Weak influence
connMap.P5.inj{3}.distance_m = 1300;

%% ========== METADATA ==========
connMap.metadata.description = 'Norne Field Connectivity Map';
connMap.metadata.date_created = datestr(now);
connMap.metadata.num_producers = 5;
connMap.metadata.num_injectors = 6;
connMap.metadata.strength_scale = 'Normalized 0-1, where 1 = strongest connection';
connMap.metadata.source = 'Derived from field connectivity diagram';

%% ========== VALIDATION ==========
fprintf('Connectivity Map Built:\n');
fprintf('  Producers: %d\n', connMap.metadata.num_producers);
fprintf('  Injectors: %d\n', connMap.metadata.num_injectors);

% Test weights for each producer
producers = {'P1', 'P2', 'P3', 'P4', 'P5'};
for i = 1:length(producers)
    p = producers{i};
    weights = getInjectorWeights(connMap, p);
    fprintf('  %s weights: [', p);
    fprintf('%.2f ', weights(1:6));
    fprintf(']\n');
    
    [~, best_inj] = max(weights);
    fprintf('    → Best injector: I%d (weight=%.2f)\n', best_inj, weights(best_inj));
end

fprintf('\nConnectivity map ready for NMPC.\n');

end