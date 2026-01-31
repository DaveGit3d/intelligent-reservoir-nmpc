function [y_pred_physical, y_pred_scaled] = predict_surrogate_model(...
    nlarx_model, u_physical, ypast_physical, upast_physical, scaling_params, verbose)
% PREDICT_SURROGATE_MODEL: Multi-step prediction for NMPC using NLARX model
%
% ✅ CRITICAL FIX: Properly initializes input lags using actual past controls
%
% INPUTS:
%   nlarx_model       : Trained idnlarx model object
%   u_physical        : Control sequence [5 x Np] in PHYSICAL units
%                       Row 1: BHP (psi)
%                       Rows 2-5: Injection rates (BBL/day) for I1, I2, I3, I4
%   ypast_physical    : Past outputs [na x 1] in PHYSICAL units (STB/day)
%   upast_physical    : Past inputs [5 x nb_max] in PHYSICAL units
%                       upast_physical(:, end) = u(t-1) (most recent)
%                       upast_physical(:, end-1) = u(t-2)
%   scaling_params    : Struct with fields: input_mean, input_std, output_mean, output_std
%   verbose           : (optional) 0=silent, 1=print debug info
%
% OUTPUTS:
%   y_pred_physical   : Predicted outputs [Np x 1] in PHYSICAL units (STB/day)
%   y_pred_scaled     : Predicted outputs [Np x 1] in SCALED units (for debugging)

if nargin < 6
    verbose = 0;
end

%% ========== EXTRACT DIMENSIONS ==========
Np = size(u_physical, 2);
n_inputs = size(u_physical, 1);
na = nlarx_model.na;
nb = nlarx_model.nb;
nb_max = max(nb);

if verbose
    fprintf('\n=== PREDICT_SURROGATE_MODEL ===\n');
    fprintf('  Prediction horizon: %d steps\n', Np);
    fprintf('  Number of inputs: %d\n', n_inputs);
    fprintf('  Model orders: na=%d, nb=[%s]\n', na, num2str(nb));
end

%% ========== VALIDATE INPUTS ==========
if n_inputs ~= 5
    error('u_physical must have 5 rows [BHP; I1; I2; I3; I4], got %d', n_inputs);
end

if length(ypast_physical) < na
    error('ypast_physical must have at least %d samples (na=%d), got %d', ...
        na, na, length(ypast_physical));
end

if size(upast_physical, 1) ~= 5
    error('upast_physical must have 5 rows (one per input), got %d', size(upast_physical, 1));
end

if size(upast_physical, 2) < nb_max
    warning('upast_physical has %d columns, but nb_max=%d. Padding with last value.', ...
        size(upast_physical, 2), nb_max);
    upast_physical = [repmat(upast_physical(:, 1), 1, nb_max - size(upast_physical, 2)), upast_physical];
end

%% ========== SCALE INPUTS ==========
% Scale future control sequence
u_scaled = zeros(size(u_physical));
for i = 1:n_inputs
    u_scaled(i, :) = (u_physical(i, :) - scaling_params.input_mean(i)) / ...
                      scaling_params.input_std(i);
end

% Scale past outputs (only use most recent na samples)
ypast_for_init = ypast_physical(end-na+1:end);
ypast_scaled = (ypast_for_init - scaling_params.output_mean) / scaling_params.output_std;

% Scale past inputs
upast_scaled = zeros(size(upast_physical));
for i = 1:n_inputs
    upast_scaled(i, :) = (upast_physical(i, :) - scaling_params.input_mean(i)) / ...
                          scaling_params.input_std(i);
end

if verbose
    fprintf('  Input range (physical): BHP=[%.0f, %.0f], Inj=[%.0f, %.0f]\n', ...
        min(u_physical(1,:)), max(u_physical(1,:)), ...
        min(u_physical(2:end,:),[],'all'), max(u_physical(2:end,:),[],'all'));
    fprintf('  Past outputs (physical): [%.0f, %.0f, ...]\n', ypast_for_init(1), ypast_for_init(min(2,end)));
end

%% ========== MULTI-STEP PREDICTION ==========
y_pred_scaled = zeros(Np, 1);
y_buffer = ypast_scaled(:);

% Ensure we have exactly 'na' past outputs
if length(y_buffer) > na
    y_buffer = y_buffer(end-na+1:end);
end

for k = 1:Np
    % Build regressor in EXACT required order: 12 elements (ROW vector)
    reg = zeros(1, 12);
    
    % Output lags: y(t-1), y(t-2)
    reg(1) = y_buffer(end);
    reg(2) = y_buffer(end-1);
    
    % Input lags with proper initialization
    reg_idx = 3;
    for i = 1:5
        % Determine u_i(t-1)
        if k == 1
            u_t1 = upast_scaled(i, end);  % Use actual past
        else
            u_t1 = u_scaled(i, k-1);
        end
        
        % Determine u_i(t-2)
        if k == 1
            if size(upast_scaled, 2) >= 2
                u_t2 = upast_scaled(i, end-1);
            else
                u_t2 = u_t1;
            end
        elseif k == 2
            u_t2 = upast_scaled(i, end);
        else
            u_t2 = u_scaled(i, k-2);
        end
        
        reg(reg_idx)     = u_t1;
        reg(reg_idx + 1) = u_t2;
        reg_idx = reg_idx + 2;
    end

    % Evaluate nonlinearity
    y_next_scaled = evaluate(nlarx_model.Nonlinearity, reg);
    y_pred_scaled(k) = y_next_scaled(1);
    
    % Update output buffer
    y_buffer = [y_buffer; y_next_scaled(1)];
    if length(y_buffer) > na
        y_buffer = y_buffer(end-na+1:end);
    end

    if verbose && (k == 1 || k == Np || mod(k,5)==0)
        y_phys = y_next_scaled(1) * scaling_params.output_std + scaling_params.output_mean;
        fprintf('    Step %d: %.0f STB/day\n', k, y_phys);
    end
end

%% ========== UNSCALE OUTPUTS ==========
y_pred_physical = y_pred_scaled * scaling_params.output_std + scaling_params.output_mean;

%% ========== VERBOSE OUTPUT ==========
if verbose
    fprintf('  Prediction results:\n');
    fprintf('    Physical: [%.0f, %.0f, ..., %.0f] STB/day\n', ...
        y_pred_physical(1), y_pred_physical(min(2,end)), y_pred_physical(end));
    fprintf('    Mean: %.0f STB/day\n', mean(y_pred_physical));
    fprintf('    Range: [%.0f, %.0f] STB/day\n', min(y_pred_physical), max(y_pred_physical));
end

%% ========== SANITY CHECKS ==========
% Check for unrealistic predictions
if any(y_pred_physical < 0)
    warning('Negative oil production predicted! Clipping to zero.');
    y_pred_physical = max(y_pred_physical, 0);
end

if any(y_pred_physical > 50000)
    warning('Unrealistically high oil production (>50k STB/day). Check model or inputs.');
end

% Check for NaN/Inf
if any(isnan(y_pred_physical)) || any(isinf(y_pred_physical))
    error('Prediction returned NaN or Inf values! Check scaling parameters and model.');
end

% Ensure output is column vector
y_pred_physical = y_pred_physical(:);
y_pred_scaled = y_pred_scaled(:);

end
