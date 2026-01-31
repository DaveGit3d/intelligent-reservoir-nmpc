%% train_narx_robust_final.m
% Train robust NLARX surrogate models for P1–P5 with anti-overfitting measures
% Designed for limited Norne data (~100 samples) and NMPC integration
% clear; clc; close all;
warning('off', 'ident:idnlmodel:computeModelQualityMetrics:TooFewData');

%% === CONFIGURATION (Reduced Complexity) ===
inputDelays = 1;            % Only 1-step input memory
feedbackDelays = 1;         % Only 1-step output feedback
hiddenLayerSize = 4;        % Small network (4 units)
testLags = [0, 1, 2, 5];    % Smaller lags → more data
nInjectorsPerProducer = 4;
sampleTime = 1;             % 1 day
trainSplit = 0.70;          % 70% train, 30% test

%% === BUILD CONNECTIVITY MAP ===
load('ntwrk.mat', 'ntwrk');
connMap = buildAdaptiveConnMap(ntwrk, nInjectorsPerProducer);

%% === LOAD INJECTOR DATA ===
injectors = unique(vertcat(connMap.P1, connMap.P2, connMap.P3, connMap.P4, connMap.P5));
injData = containers.Map;
for j = 1:numel(injectors)
    fname = [injectors{j} '.csv'];
    if isfile(fname)
        injData(injectors{j}) = readtable(fname);
    else
        error('Missing injector file: %s', fname);
    end
end

producers = fieldnames(connMap);
results = struct();

%% === TRAIN EACH PRODUCER ===
for k = 1:numel(producers)
    well = producers{k};
    fprintf('\n=== Training NLARX for %s ===\n', well);
    
    if ~isfile([well '.csv'])
        warning('Missing %s.csv. Skipping.', well);
        continue;
    end
    prod = readtable([well '.csv']);
    
    bestTestR2 = -inf;
    bestModel = [];
    bestLag = NaN;
    bestRes = struct();
    bestScaler = struct();
    injList = connMap.(well);
    
    for lag = testLags
        fprintf('   >> Testing lag = %d\n', lag);
        
        % Merge data
        T = prod(:, {'Time','BHP','OilRate'});
        for j = 1:numel(injList)
            injTbl = injData(injList{j});
            T = innerjoin(T, injTbl, 'Keys', 'Time');
            T.Properties.VariableNames{end} = [injList{j} '_Rate'];
        end
        
        % Apply lag
        for j = 1:numel(injList)
            col = [injList{j} '_Rate'];
            T.(col) = [nan(lag,1); T.(col)(1:end-lag)];
        end
        T = rmmissing(T);
        
        if size(T,1) < 50
            warning('   ⚠️ Too few samples (%d). Skipping lag=%d.', size(T,1), lag);
            continue;
        end
        
        %% === TRAIN/TEST SPLIT ===
        n = size(T,1);
        trainSize = floor(trainSplit * n);
        trainIdx = 1:trainSize;
        testIdx = (trainSize+1):n;
        
        T_train = T(trainIdx, :);
        T_test = T(testIdx, :);
        
        %% === NORMALIZE (on train only) ===
        muX = mean(T_train{:, 2:end-1});
        sigmaX = std(T_train{:, 2:end-1});
        muY = mean(T_train.OilRate);
        sigmaY = std(T_train.OilRate);
        
        sigmaX(sigmaX < 1e-6) = 1;
        if sigmaY < 1e-6, sigmaY = 1; end
        
        X_train = (T_train{:, 2:end-1} - muX) ./ sigmaX;
        Y_train = (T_train.OilRate - muY) ./ sigmaY;
        X_test = (T_test{:, 2:end-1} - muX) ./ sigmaX;
        Y_test = (T_test.OilRate - muY) / sigmaY;
        
        %% === BUILD MODEL ===
        na = feedbackDelays;
        nb = inputDelays * ones(1, size(X_train,2));
        nk = ones(1, size(X_train,2));
        
        idTrain = iddata(Y_train, X_train, sampleTime);
     
            sys = nlarx(idTrain, [na nb nk], idSigmoidNetwork(hiddenLayerSize));
        
        
        %% === EVALUATE ===
        Yhat_train = predict(sys, idTrain);
        Yhat_test = predict(sys, iddata(Y_test, X_test, sampleTime));
        
        % Denormalize
        y_true_train = Y_train * sigmaY + muY;
        y_pred_train = Yhat_train.OutputData * sigmaY + muY;
        y_true_test = Y_test * sigmaY + muY;
        y_pred_test = Yhat_test.OutputData * sigmaY + muY;
        
        % Metrics
        SS_res = sum((y_true_test - y_pred_test).^2);
        SS_tot = sum((y_true_test - mean(y_true_test)).^2);
        R2_test = 1 - SS_res/SS_tot;
        fit_test = max(0, R2_test * 100);
        
        fprintf('      Test R²: %.4f | Fit: %.2f%%\n', R2_test, fit_test);
        
      %% === TRACK BEST ===
if R2_test > bestTestR2
    bestTestR2 = R2_test;
    bestLag = lag;
    bestModel = sys;
    bestRes.test_actual = y_true_test;
    bestRes.test_predicted = y_pred_test;
    bestRes.test_time = T_test.Time;
    bestRes.test_inputs = T_test{:, 2:end-1};  % ✅ CRITICAL ADDITION
    bestRes.train_actual = y_true_train;
    bestRes.train_predicted = y_pred_train;
    bestRes.train_time = T_train.Time;
    bestScaler.muX = muX;
    bestScaler.sigmaX = sigmaX;
    bestScaler.muY = muY;
    bestScaler.sigmaY = sigmaY;
end
    end
    
    if isempty(bestModel)
        warning('❌ No valid model for %s.', well);
        continue;
    end
    
   %% === PLOT & SAVE SEPARATELY ===
fit_test = max(0, bestTestR2 * 100);  % Fitting percentage

% Create folder if it doesn't exist
plotDir = 'training_plots';
if ~exist(plotDir, 'dir')
    mkdir(plotDir);
end

wellName = well;

% --- 1. Training Time-Series ---
fig1 = figure('Position', [100, 100, 800, 400]);
plot(bestRes.train_time, bestRes.train_actual, 'b', 'LineWidth', 1.2); hold on;
plot(bestRes.train_time, bestRes.train_predicted, 'r--', 'LineWidth', 1.2);
title(sprintf('%s - Training Data (Lag = %d)', wellName, bestLag));
xlabel('Time'); ylabel('Oil Rate [bbl/day]');
legend('Actual', 'Predicted', 'Location', 'best');
grid on; box on;
plotFile1 = fullfile(plotDir, sprintf('%s_training.png', wellName));
print(fig1, '-dpng', plotFile1, '-r300');
close(fig1);

% --- 2. Test Time-Series ---
fig2 = figure('Position', [100, 100, 800, 400]);
plot(bestRes.test_time, bestRes.test_actual, 'b', 'LineWidth', 1.2); hold on;
plot(bestRes.test_time, bestRes.test_predicted, 'r--', 'LineWidth', 1.2);
title(sprintf('%s - Test Data (R^2 = %.3f, Fit = %.1f%%)', wellName, bestTestR2, fit_test));
xlabel('Time'); ylabel('Oil Rate [bbl/day]');
legend('Actual', 'Predicted', 'Location', 'best');
grid on; box on;
plotFile2 = fullfile(plotDir, sprintf('%s_test.png', wellName));
print(fig2, '-dpng', plotFile2, '-r300');
close(fig2);

% --- 3. Parity Plot (Actual vs Predicted) ---
fig3 = figure('Position', [100, 100, 600, 600]);
scatter(bestRes.test_actual, bestRes.test_predicted, 30, 'filled', 'b');
hold on;
lims = [min([bestRes.test_actual; bestRes.test_predicted]), ...
        max([bestRes.test_actual; bestRes.test_predicted])];
plot(lims, lims, 'r--', 'LineWidth', 1.5);
xlabel('Actual Oil Rate [bbl/day]');
ylabel('Predicted Oil Rate [bbl/day]');
title(sprintf('%s - Parity Plot (Test Set)', wellName));
grid on; axis equal; box on;
plotFile3 = fullfile(plotDir, sprintf('%s_parity.png', wellName));
print(fig3, '-dpng', plotFile3, '-r300');
close(fig3);

% --- 4. Summary Metric Plot (R² + Fit % as text) ---
fig4 = figure('Position', [100, 100, 500, 300]);
clf; axis off;
text(0.1, 0.7, sprintf('Well: %s', wellName), 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.5, sprintf('Best Lag: %d', bestLag), 'FontSize', 12);
text(0.1, 0.3, sprintf('Test R^2: %.4f', bestTestR2), 'FontSize', 12);
text(0.1, 0.1, sprintf('Fitting Accuracy: %.2f%%', fit_test), 'FontSize', 12, 'Color', 'b');
title('Model Performance Summary', 'FontSize', 13);
plotFile4 = fullfile(plotDir, sprintf('%s_summary.png', wellName));
print(fig4, '-dpng', plotFile4, '-r300');
close(fig4);

fprintf('📊 Saved plots for %s:\n', wellName);
fprintf('   - Training: %s\n', plotFile1);
fprintf('   - Test:     %s\n', plotFile2);
fprintf('   - Parity:   %s\n', plotFile3);
fprintf('   - Summary:  %s\n', plotFile4);

    % fprintf('📊 Saved training plot: %s\n', plotFile);

    % --- AFTER you selected bestModel & bestScaler ---
    % Ensure there is a 'sys' variable (for older code compatibility)
    sys = bestModel;
    
    % Save canonical model file (contains sys and bestModel)
    modelFile = sprintf('%s_NLARX_Model_best.mat', well);
    save(modelFile, 'bestModel', 'sys', 'bestLag', 'bestTestR2', ...
         'muX', 'sigmaX', 'muY', 'sigmaY', 'inputDelays', 'feedbackDelays');
    
    % Save Simulink-friendly file (contains net, modelType, scaler, XSigma, YSigma)
    simulinkFile = sprintf('%s_NLARX_Simulink.mat', well);
    net = bestModel;                   % keep naming consistent
    modelType = 'nlarx';
    scaler.meanX = bestScaler.muX(:);
    scaler.stdX  = bestScaler.sigmaX(:);
    scaler.meanY = bestScaler.muY(:);
    scaler.stdY  = bestScaler.sigmaY(:);
    XSigma = bestScaler.sigmaX;
    YSigma = bestScaler.sigmaY;
    validation.R2_test = bestTestR2;
    save(simulinkFile, 'net', 'modelType', 'scaler', 'XSigma', 'YSigma', 'bestLag', 'validation', '-v7.3');
    
    fprintf('✅ Saved: %s and %s\n', modelFile, simulinkFile);
end

fprintf('\n✅ Training complete. Use *_NLARX_Simulink.mat in NMPC.\n');