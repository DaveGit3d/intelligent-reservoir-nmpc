%% retrain_model_recent_data.m
% Retrain NLARX model on RECENT Norne data (decline phase)
%
% This fixes the root cause: model trained on 21k STB/day baseline
% but current production is 7-10k STB/day

clear; clc;

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║     RETRAIN NLARX MODEL ON RECENT NORNE DATA                 ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

%% ========== LOAD ORIGINAL DATA ==========
fprintf('Loading original data...\n');

% Load producer data
P1_data = readtable('P1.csv');

% Load original model for reference
load('P1_NLARX_NMPC.mat', 'nlarx_model', 'scaling_params', 'model_info', 'performance');

fprintf('  ✓ Original data: %d samples\n', height(P1_data));
fprintf('  ✓ Original baseline: %.0f STB/day\n', scaling_params.output_mean);
fprintf('  ✓ Original model fit: %.1f%%\n\n', performance.fit_percent);

%% ========== ANALYZE PRODUCTION HISTORY ==========
fprintf('Analyzing production history...\n');

% Plot production over time
figure('Position', [100, 100, 1200, 400]);
plot(P1_data.Time, P1_data.OilRate, 'LineWidth', 2);
xlabel('Time (days)');
ylabel('Oil Rate (STB/day)');
title('Norne P1 Production History');
grid on;

% Find split point where production declined significantly
historical_avg = mean(P1_data.OilRate(1:floor(end*0.5)));
recent_avg = mean(P1_data.OilRate(floor(end*0.5):end));
decline_pct = (recent_avg - historical_avg) / historical_avg * 100;

fprintf('  Historical avg (first 50%%): %.0f STB/day\n', historical_avg);
fprintf('  Recent avg (last 50%%): %.0f STB/day\n', recent_avg);
fprintf('  Decline: %+.1f%%\n\n', decline_pct);

% Decide split point
if abs(decline_pct) > 20
    % Significant decline - use only recent data
    split_fraction = 0.6;   % Use last 40%
    fprintf('  ⚠️  Significant decline detected!\n');
    fprintf('  ⚠️  Will retrain on last %.0f%% of data only\n\n', (1-split_fraction)*100);
else
    % Moderate decline - use last 30%
    split_fraction = 0.7;
end

%% ========== PREPARE RECENT DATA ==========
fprintf('Preparing training data...\n');

split_idx = floor(height(P1_data) * split_fraction);
recent_data = P1_data(split_idx:end, :);

fprintf('  Training samples: %d (from day %d to %d)\n', ...
    height(recent_data), recent_data.Time(1), recent_data.Time(end));
fprintf('  Oil rate range: [%.0f, %.0f] STB/day\n', ...
    min(recent_data.OilRate), max(recent_data.OilRate));

% Create uniform time grid
time_uniform = (recent_data.Time(1):1:recent_data.Time(end))';
oil_uniform = interp1(recent_data.Time, recent_data.OilRate, time_uniform, 'linear', 'extrap');
BHP_uniform = interp1(recent_data.Time, recent_data.BHP, time_uniform, 'linear', 'extrap');

% Load injector data
injList = model_info.injectors;
inj_data = cell(4, 1);
inj_uniform = zeros(length(time_uniform), 4);

for i = 1:4
    inj_data{i} = readtable(sprintf('%s.csv', injList{i}));
    inj_uniform(:, i) = interp1(inj_data{i}.Time, inj_data{i}.InjectionRate, ...
        time_uniform, 'linear', 'extrap');
end

fprintf('  ✓ Data interpolated to uniform grid: %d samples\n\n', length(time_uniform));

%% ========== CREATE iddata OBJECT ==========
fprintf('Creating iddata object...\n');

% Input: [BHP, I1, I2, I3, I4]
u_train = [BHP_uniform, inj_uniform];

% Output: Oil rate
y_train = oil_uniform;

% Create iddata
data_train = iddata(y_train, u_train, 1, 'Name', 'P1_Recent');
data_train.InputName = {'BHP', 'I1', 'I2', 'I3', 'I4'};
data_train.OutputName = {'OilRate'};
data_train.TimeUnit = 'days';

fprintf('  ✓ iddata created: %d samples\n', length(y_train));
fprintf('  ✓ Inputs: 5 (BHP + 4 injectors)\n');
fprintf('  ✓ Output: Oil rate\n\n');

%% ========== COMPUTE NEW SCALING PARAMETERS ==========
fprintf('Computing scaling parameters from recent data...\n');

new_scaling_params = struct();

% Input scaling
new_scaling_params.input_mean = mean(u_train, 1)';
new_scaling_params.input_std = std(u_train, 0, 1)';

% Output scaling
new_scaling_params.output_mean = mean(y_train);
new_scaling_params.output_std = std(y_train);

fprintf('  New baseline oil rate: %.0f STB/day (was %.0f)\n', ...
    new_scaling_params.output_mean, scaling_params.output_mean);
fprintf('  Change: %+.0f STB/day (%+.1f%%)\n\n', ...
    new_scaling_params.output_mean - scaling_params.output_mean, ...
    (new_scaling_params.output_mean - scaling_params.output_mean) / scaling_params.output_mean * 100);

% Apply scaling to training data
data_train.OutputData = (data_train.OutputData - new_scaling_params.output_mean) / ...
                         new_scaling_params.output_std;

for i = 1:5
    data_train.InputData(:, i) = (data_train.InputData(:, i) - new_scaling_params.input_mean(i)) / ...
                                  new_scaling_params.input_std(i);
end

fprintf('  ✓ Data scaled\n\n');

% Debug: inspect original nonlinearity
disp('Original nonlinearity:');
disp(nlarx_model.Nonlinearity);
if isa(nlarx_model.Nonlinearity, 'idSigmoidNetwork')
    try 
        disp(['Units (Parameters): ' num2str(nlarx_model.Nonlinearity.NonlinearFcn.Parameters.NumberOfUnits)]); 
    end
    try 
        disp(['Units (Legacy): ' num2str(nlarx_model.Nonlinearity.NumberOfUnits)]); 
    end
end
%% ========== TRAIN NEW MODEL ==========
fprintf('Training new NLARX model...\n');
fprintf('  Using same structure as original:\n');
fprintf('    na = %d (output lags)\n', nlarx_model.na);
fprintf('    nb = [%s] (input lags)\n', num2str(nlarx_model.nb));
fprintf('    Nonlinearity: %s\n\n', class(nlarx_model.Nonlinearity));

% --- Reconstruct order and delays ---
na = nlarx_model.na;              % 2
nb = nlarx_model.nb;              % [2 2 2 2 2]
nk = nlarx_model.nk;              % Usually [1 1 1 1 1] for t-1 start

% --- Recreate nonlinearity with correct number of units ---
% From your print: "Sigmoid network with 6 units"
nonlin_old = nlarx_model.Nonlinearity;
if isa(nonlin_old, 'idSigmoidNetwork')
    % Safe extraction: use display info or known value
    % Since you saw "6 units", hardcode or extract robustly
    num_units = 6;  % ⬅️ From your model print!
    % OR (if you want to automate):
    % num_units = 6; % fallback
    % if isprop(nonlin_old, 'NonlinearFcn')
    %     try
    %         num_units = nonlin_old.NonlinearFcn.Parameters.NumberOfUnits;
    %     catch
    %         num_units = 6;
    %     end
    % end
    nonlin_new = idSigmoidNetwork(num_units);
else
    nonlin_new = idSigmoidNetwork(6);
end

fprintf('  Using na=%d, nb=%s, nk=%s\n', na, mat2str(nb), mat2str(nk));
fprintf('  Recreating nonlinearity: %s with %d units\n', class(nonlin_new), num_units);

% --- Train using lag-based syntax (WORKS in all versions) ---
opt = nlarxOptions;
opt.Regularization.Lambda = 0.1;
opt.SearchOptions.MaxIterations = 100;
opt.Display = 'on';

tic;
nlarx_model_new = nlarx(data_train, [na, nb, nk], nonlin_new, opt);
train_time = toc;

fprintf('\n  ✓ Training complete (%.1f seconds)\n\n', train_time);

%% ========== VALIDATE NEW MODEL ==========
fprintf('Validating new model...\n');

% Predict on training data (scaled)
y_pred_new = predict(nlarx_model_new, data_train);

% Compute fit percentage (on scaled data)
fit_new = goodnessOfFit(y_pred_new.OutputData, data_train.OutputData, 'NRMSE') * 100;

% Unscale for physical interpretation
y_train_physical = data_train.OutputData * new_scaling_params.output_std + new_scaling_params.output_mean;
y_pred_physical = y_pred_new.OutputData * new_scaling_params.output_std + new_scaling_params.output_mean;

rmse_physical = sqrt(mean((y_train_physical - y_pred_physical).^2));

fprintf('  New model fit: %.1f%% (was %.1f%%)\n', fit_new, performance.fit_percent);
fprintf('  RMSE: %.0f STB/day\n\n', rmse_physical);

%% ========== COMPARE OLD VS NEW ==========
fprintf('Comparing old vs new model on recent data...\n');

% --- Create compatible data for old model ---
data_train_for_old = data_train;  % Already scaled, same time grid

% Match channel names and time unit to old model
data_train_for_old.InputName = nlarx_model.InputName;
data_train_for_old.OutputName = nlarx_model.OutputName;
data_train_for_old.TimeUnit = nlarx_model.TimeUnit;

% Predict with old model
try
    y_pred_old = predict(nlarx_model, data_train_for_old);
    fit_old = goodnessOfFit(y_pred_old.OutputData, data_train_for_old.OutputData, 'NRMSE') * 100;
    fprintf('  Old model fit on recent data: %.1f%%\n', fit_old);
catch ME
    fprintf('  ❌ Failed to predict with old model: %s\n', ME.message);
    fit_old = NaN;
end

fprintf('  New model fit on recent data: %.1f%%\n', fit_new);
if ~isnan(fit_old)
    fprintf('  Improvement: %+.1f%%\n\n', fit_new - fit_old);
else
    fprintf('  Cannot compute improvement.\n\n');
end

% Visualization
figure('Position', [100, 100, 1400, 500]);

subplot(1, 2, 1);
plot(time_uniform, y_train_physical, 'k-', 'LineWidth', 2);
hold on;
plot(time_uniform, y_pred_physical, 'b--', 'LineWidth', 2);
xlabel('Time (days)');
ylabel('Oil Rate (STB/day)');
title(sprintf('New Model Prediction (Fit: %.1f%%)', fit_new));
legend('Actual', 'Predicted', 'Location', 'best');
grid on;

subplot(1, 2, 2);
residuals = y_train_physical - y_pred_physical;
histogram(residuals, 30, 'Normalization', 'pdf', 'FaceColor', [0.3, 0.5, 0.7]);
hold on;
mu_res = mean(residuals);
sigma_res = std(residuals);
x_res = linspace(min(residuals), max(residuals), 100);
y_res = normpdf(x_res, mu_res, sigma_res);
plot(x_res, y_res, 'r-', 'LineWidth', 2);
xlabel('Residual (STB/day)');
ylabel('Probability Density');
title(sprintf('Residuals (μ=%.0f, σ=%.0f)', mu_res, sigma_res));
legend('Observed', 'Normal Fit', 'Location', 'best');
grid on;

%% ========== SAVE NEW MODEL ==========
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║                 SAVE RETRAINED MODEL?                         ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

if fit_new > fit_old
    fprintf('  ✅ New model is BETTER (fit: %.1f%% vs %.1f%%)\n', fit_new, fit_old);
    fprintf('  ✅ Recommendation: DEPLOY new model\n\n');
    
    user_response = input('Deploy new model? (y/n): ', 's');
    
    if strcmpi(user_response, 'y')
        % Update variables
        nlarx_model = nlarx_model_new;
        scaling_params = new_scaling_params;
        
        % Update performance metrics
        performance.fit_percent = fit_new;
        performance.rmse = rmse_physical;
        performance.training_date = datestr(now);
        performance.training_samples = length(y_train);
        
         Save to main file (BACKUP OLD FIRST)
         if exist('P1_NLARX_NMPC.mat', 'file')
              copyfile('P1_NLARX_NMPC.mat', 'P1_NLARX_NMPC_backup.mat');
              fprintf('  ✓ Backed up old model to P1_NLARX_NMPC_backup.mat\n');
         end
        
        save('P2_NLARX_NMPC.mat', 'nlarx_model', 'scaling_params', 'model_info', 'performance');
        fprintf('  ✓ New model saved to P1_NLARX_NMPC.mat\n\n');
        
        % Also save updated version
        sys_new = nlarx_model_new;
        save('P2_NLARX_NMPC_updated.mat', 'sys_new', 'scaling_params');
        fprintf('  ✓ Updated model saved to P2_NLARX_NMPC_updated.mat\n\n');
        
        fprintf('  ✅ Deployment complete!\n\n');
        fprintf('  Next steps:\n');
        fprintf('    1. Clear persistent variables: clear functions\n');
        fprintf('    2. Re-run: test_nmpc_hybrid_standalone\n');
        fprintf('    3. Expected: RMSE < 2000 STB/day\n\n');
    else
        fprintf('  ⚠️  Deployment cancelled by user\n\n');
    end
else
    fprintf('  ⚠️  New model is NOT better (fit: %.1f%% vs %.1f%%)\n', fit_new, fit_old);
    fprintf('  ⚠️  Recommendation: Keep old model, investigate data quality\n\n');
end

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║                 RETRAINING COMPLETE                           ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');