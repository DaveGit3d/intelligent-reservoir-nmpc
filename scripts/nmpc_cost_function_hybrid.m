function J = nmpc_cost_function_hybrid(u_vec, Np, Nu, ypast, upast, model, ...
    scaling_params, weights, constraints, bias_correction, injector_weights)
% NMPC_COST_FUNCTION_HYBRID - Connectivity-weighted economic cost function
%
% FORMULATION:
%   J = Σ[-p_oil*q_oil + Σ(c_water_i * q_water_i) + λ_u||Δu||² + λ_alloc*penalty]
%
% where c_water_i = base_cost * (1 / connectivity_weight_i)^α
%
% INPUTS:
%   u_vec            - Control sequence [BHP; inj1-4] × Nu (5*Nu × 1)
%   Np               - Prediction horizon
%   Nu               - Control horizon
%   ypast            - Past oil rates for NLARX
%   upast            - Past controls for NLARX
%   model            - NLARX surrogate model
%   scaling_params   - Model scaling parameters
%   weights          - Economic weights (oil_production, water_injection, control_effort)
%   constraints      - Bounds and targets
%   bias_correction  - Prediction bias correction
%   injector_weights - Connectivity weights [4×1] (CRITICAL)
%
% OUTPUTS:
%   J - Total cost (minimize)

%% Extract Parameters
p_oil = weights.oil_production;        % $/STB
c_water_base = weights.water_injection; % $/BBL (base cost)
lambda_u = weights.control_effort;      % Control smoothness

% Reshape control sequence
u_seq = reshape(u_vec, 5, Nu);  % [5 × Nu]

% Extend control horizon to prediction horizon
if Nu < Np
    u_full = [u_seq, repmat(u_seq(:, end), 1, Np - Nu)];
else
    u_full = u_seq;
end

%% Predict Oil Production
try
    [y_pred, ~] = predict_surrogate_model(model, u_full, ypast, upast, scaling_params, 0);
    y_pred = y_pred + bias_correction;  % Apply bias correction
catch ME
    warning('Cost:PredictionFailed', 'Model prediction failed: %s', ME.message);
    J = 1e12;  % Return large cost on failure
    return;
end

%% ========== NORMALIZE CONNECTIVITY WEIGHTS ==========
injector_weights = injector_weights(:);
weight_sum = sum(injector_weights);
if weight_sum < 1e-6
    norm_weights = ones(4, 1) / 4;
else
    norm_weights = injector_weights / weight_sum;
end

%% ========== PER-INJECTOR WATER COST MULTIPLIERS ==========
% Make low-connectivity injectors more expensive
alpha = 1.2;  % Exponent for cost scaling (1.0-2.0 range)
% Cost multiplier = (1 / normalized_weight)^alpha
cost_multipliers = zeros(4, 1);
for i = 1:4
    if norm_weights(i) > 0.01
        cost_multipliers(i) = (1 / norm_weights(i))^alpha;
    else
        cost_multipliers(i) = 10;  % Very high cost for near-zero connectivity
    end
end

% Normalize to preserve average cost
avg_multiplier = mean(cost_multipliers);
cost_multipliers = cost_multipliers / avg_multiplier;

%% ========== APPLY VARIABLE WATER COSTS ==========
water_costs = zeros(Np, 1);
for k = 1:Np
    for i = 1:4
        % Variable cost per injector based on connectivity
        water_costs(k) = water_costs(k) + ...
            c_water_base * cost_multipliers(i) * u_full(i+1, k);
    end
end

%% Revenue from Oil Production
oil_revenue = p_oil * y_pred;  % [Np × 1]

%% Control Effort Penalty (Smoothness)
control_penalty = 0;
for k = 1:Nu-1
    du = u_seq(:, k+1) - u_seq(:, k);
    control_penalty = control_penalty + lambda_u * sum(du.^2);
end

%% ========== SOFT CONNECTIVITY ALLOCATION PENALTY (ENHANCED) ==========
% Penalize deviations from connectivity-weighted proportions
lambda_allocation = 30000;  % Increased from 5000 (tunable: 20000-50000)
allocation_penalty = 0;

for k = 1:Nu
    % Total injection at time k
    total_inj_k = sum(u_seq(2:5, k));
    
    % Skip if near zero injection (avoid division issues)
    if total_inj_k < 1000
        continue;
    end
    
    % Compute squared deviations from connectivity-weighted target
    for i = 1:4
        % Expected injection based on connectivity
        expected_inj = norm_weights(i) * total_inj_k;
        
        % Actual injection
        actual_inj = u_seq(i+1, k);
        
        % Relative deviation (normalized by total)
        deviation = (actual_inj - expected_inj) / total_inj_k;
        
        % Quadratic penalty with connectivity weighting
        % Higher penalty for deviating from high-connectivity wells
        connectivity_emphasis = 1.0 + 0.5 * norm_weights(i);  % [1.0, 1.5]
        
        allocation_penalty = allocation_penalty + ...
            lambda_allocation * connectivity_emphasis * total_inj_k * deviation^2;
    end
end

%% ========== MILD CONNECTIVITY BONUS (OPTIONAL) ==========
% Provides gradient even when constraints are symmetric
connectivity_bonus = 0;
if lambda_allocation > 0
    lambda_connectivity = 0.05 * p_oil;  % 5% of oil price (≈$4/STB)
    
    for k = 1:Nu
        for i = 1:4
            % Small reward for allocating to high-connectivity wells
            connectivity_bonus = connectivity_bonus + ...
                lambda_connectivity * norm_weights(i) * u_seq(i+1, k) / 1000;
        end
    end
end

%% Pressure Maintenance Penalty (Optional)
if isfield(constraints, 'BHP_target') && isfield(weights, 'pressure_maintenance')
    BHP_target = constraints.BHP_target;
    lambda_p = weights.pressure_maintenance;
    
    pressure_penalty = 0;
    for k = 1:Nu
        pressure_penalty = pressure_penalty + ...
            lambda_p * (u_seq(1, k) - BHP_target)^2;
    end
else
    pressure_penalty = 0;
end

%% VRR Soft Constraint Penalty
if isfield(constraints, 'dcs_VRR_target')
    VRR_target = constraints.dcs_VRR_target;
    VRR_acceptable = get_field(constraints, 'dcs_VRR_acceptable', VRR_target * 1.5);
    
    lambda_vrr = 1e4;  % VRR penalty weight
    
    vrr_penalty = 0;
    for k = 1:Np
        total_inj = sum(u_full(2:5, k));
        oil_k = max(y_pred(k), 1e-3);
        VRR_k = total_inj / oil_k;
        
        % Soft constraint: penalize VRR > acceptable
        if VRR_k > VRR_acceptable
            vrr_penalty = vrr_penalty + lambda_vrr * (VRR_k - VRR_acceptable)^2;
        % Light penalty for VRR < target (encourage some injection)
        elseif VRR_k < VRR_target * 0.5
            vrr_penalty = vrr_penalty + 0.1 * lambda_vrr * (VRR_target * 0.5 - VRR_k)^2;
        end
    end
else
    vrr_penalty = 0;
end

%% ========== TOTAL COST ==========
% Minimize: -revenue + costs + penalties - bonus
J = -sum(oil_revenue) + ...        % Maximize oil revenue
    sum(water_costs) + ...          % Minimize connectivity-weighted water cost
    control_penalty + ...           % Minimize control changes
    allocation_penalty + ...        % Soft connectivity enforcement (ENHANCED)
    pressure_penalty + ...          % Maintain target pressure
    vrr_penalty - ...               % Soft VRR constraint
    connectivity_bonus;             % Mild connectivity preference (NEW)

%% Numerical Stability Check
if ~isfinite(J) || isnan(J)
    J = 1e12;
end

%% Debug Output (First Call Only)
persistent penalty_debug_shown;
if isempty(penalty_debug_shown)
    penalty_debug_shown = false;
end

if ~penalty_debug_shown && allocation_penalty > 0
    penalty_debug_shown = true;
    fprintf('[Cost Debug] Connectivity weights: [%.3f, %.3f, %.3f, %.3f]\n', norm_weights);
    fprintf('[Cost Debug] Cost multipliers: [%.2f, %.2f, %.2f, %.2f]\n', cost_multipliers);
    fprintf('[Cost Debug] Allocation penalty weight: %.0f\n', lambda_allocation);
    fprintf('[Cost Debug] Allocation penalty value: %.2e\n', allocation_penalty);
    if connectivity_bonus > 0
        fprintf('[Cost Debug] Connectivity bonus: %.2e\n', connectivity_bonus);
    end
end

end

%% Helper Function
function val = get_field(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end