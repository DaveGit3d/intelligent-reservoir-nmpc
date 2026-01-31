function [u_opt, diagnostics] = economic_mpc_fallback(state, constraints, model, config)
% ECONOMIC_MPC_FALLBACK - Single-step connectivity-aware economic optimization
%
% FORMULATION:
%   maximize: p_oil*q_oil - Σ(c_i * inj_i) - λ||Δu||²
%   where c_i = base_cost * (1/weight_i)^α  (connectivity-weighted cost)
%
% BHP UNITS: bar (native units, no psi conversion)
% Historical baseline: 100 bar (±5 bar)
%
% INPUTS:
%   state       - struct(BHP(bar), inj_rates, oil_rate, injector_weights, ...)
%   constraints - struct(inj_min, inj_max, BHP_min(bar), BHP_max(bar), VRR_target, ...)
%   model       - struct(nlarx_model, scaling_params)
%   config      - struct(oil_price, base_water_cost, control_weight, ...)
%
% OUTPUTS:
%   u_opt       - Optimal control [BHP(bar); inj1-4] (5×1)
%   diagnostics - Optimization statistics

%% Extract Inputs
BHP_current = state.BHP;  % bar
inj_current = state.inj_rates(:);
oil_current = state.oil_rate;
injector_weights = state.injector_weights(:);

% Economic parameters
p_oil = get_field(config, 'oil_price', 80);
c_water_base = get_field(config, 'base_water_cost', 5);
lambda_u = get_field(config, 'control_weight', 100);
lambda_vrr = get_field(config, 'VRR_penalty', 5e4);
lambda_bhp = get_field(config, 'BHP_penalty', 1e3);

%% Connectivity-Weighted Water Costs
% Normalize weights
weight_sum = sum(injector_weights);
if weight_sum < 1e-6
    norm_weights = ones(4, 1) / 4;
else
    norm_weights = injector_weights / weight_sum;
end

% Compute per-injector cost multipliers
% Formula: cost_i = base_cost * (1/weight_i)^α
alpha = 1.5;  % Exponent (higher = more penalty for weak wells)
cost_multipliers = zeros(4, 1);
for i = 1:4
    if norm_weights(i) > 0.01
        cost_multipliers(i) = (1 / norm_weights(i))^alpha;
    else
        cost_multipliers(i) = 10;  % Very high for near-zero connectivity
    end
end

% Debug output (first 3 calls only)
persistent econ_call_count;
if isempty(econ_call_count)
    econ_call_count = 0;
end
econ_call_count = econ_call_count + 1;

if econ_call_count <= 3
    fprintf('[EconMPC Call %d] Connectivity weights: [%.3f, %.3f, %.3f, %.3f]\n', ...
        econ_call_count, norm_weights);
    fprintf('[EconMPC Call %d] Cost multipliers: [%.2fx, %.2fx, %.2fx, %.2fx]\n', ...
        econ_call_count, cost_multipliers);
    fprintf('[EconMPC Call %d] BHP: %.1f bar (min: %.1f, max: %.1f)\n', ...
        econ_call_count, BHP_current, constraints.BHP_min, constraints.BHP_max);
end

% Normalize to preserve average cost
cost_multipliers = cost_multipliers / mean(cost_multipliers);

%% Bounds (BHP in bar)
BHP_min = constraints.BHP_min;  % bar
BHP_max = constraints.BHP_max;  % bar
inj_min = constraints.inj_min(:);
inj_max = constraints.inj_max(:);

% Combined bounds for u = [BHP(bar); inj1-4]
lb = [BHP_min; inj_min];
ub = [BHP_max; inj_max];

%% Initial Guess (Connectivity-Weighted, BHP in bar)
% Start with connectivity-proportional allocation
if isfield(constraints, 'max_total_injection')
    total_inj_target = min(sum(inj_max), constraints.max_total_injection);
else
    total_inj_target = sum(inj_current);
end

inj_guess = zeros(4, 1);
for i = 1:4
    inj_guess(i) = norm_weights(i) * total_inj_target;
    inj_guess(i) = max(inj_min(i), min(inj_guess(i), inj_max(i)));
end

% BHP initial guess (bar)
BHP_target_guess = get_field(constraints, 'BHP_target', 100);  % bar (historical baseline)
BHP_guess = (BHP_current + BHP_target_guess) / 2;
u0 = [BHP_guess; inj_guess];

%% Cost Function
VRR_target = get_field(constraints, 'VRR_target', 2.5);
VRR_acceptable = get_field(constraints, 'VRR_acceptable_max', 4.0);
BHP_target = get_field(constraints, 'BHP_target', 100);  % bar

cost_fun = @(u) economic_objective(u, state, model, p_oil, ...
    c_water_base, cost_multipliers, lambda_u, lambda_vrr, lambda_bhp, ...
    VRR_target, VRR_acceptable, BHP_target);

%% Linear Constraints
A = [];
b = [];

% Total injection limit
if isfield(constraints, 'max_total_injection') && constraints.max_total_injection > 0
    A = [0, 1, 1, 1, 1];  % sum(inj) <= max
    b = constraints.max_total_injection;
end

%% Optimize
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
    'MaxIterations', 100, 'MaxFunctionEvaluations', 500, ...
    'OptimalityTolerance', 1e-3, 'StepTolerance', 1e-4);

tic;
[u_opt, fval, exitflag, output] = fmincon(cost_fun, u0, A, b, [], [], lb, ub, [], options);
opt_time = toc;

%% Predict Output with Optimal Control
try
    if isfield(state, 'oil_history') && isfield(state, 'upast')
        [oil_pred, ~] = predict_surrogate_model(model.nlarx_model, u_opt, ...
            state.oil_history(end-model.nlarx_model.na+1:end), ...
            state.upast, model.scaling_params, 0);
    else
        % Fallback: use current rate with small adjustment
        % Sensitivity: ~1% oil change per 10 bar BHP change (approximate)
        bhp_change_bar = u_opt(1) - BHP_current;
        oil_pred = oil_current * (1 + 0.001 * bhp_change_bar);
    end
catch
    oil_pred = oil_current;
end

%% Diagnostics
total_inj = sum(u_opt(2:5));
VRR_pred = total_inj / (oil_pred + 1e-3);

% Compute effective water cost
water_cost_effective = 0;
for i = 1:4
    water_cost_effective = water_cost_effective + ...
        c_water_base * cost_multipliers(i) * u_opt(i+1);
end
water_cost_effective = water_cost_effective / (total_inj + 1e-3);

diagnostics = struct();
diagnostics.exitflag = exitflag;
diagnostics.iterations = output.iterations;
diagnostics.funcCount = output.funcCount;
diagnostics.opt_time = opt_time;
diagnostics.objective_value = fval;
diagnostics.oil_predicted = oil_pred;
diagnostics.VRR_predicted = VRR_pred;
diagnostics.water_cost = water_cost_effective;
diagnostics.connectivity_weights = norm_weights;
diagnostics.cost_multipliers = cost_multipliers;
diagnostics.BHP_bar = u_opt(1);  % Store BHP in bar for reference

end

%% ========== OBJECTIVE FUNCTION (BHP IN BAR) ==========
function J = economic_objective(u, state, model, p_oil, c_water_base, ...
    cost_multipliers, lambda_u, lambda_vrr, lambda_bhp, ...
    VRR_target, VRR_acceptable, BHP_target)

BHP = u(1);  % bar
inj = u(2:5);

%% Predict Oil Production
try
    if isfield(state, 'oil_history') && isfield(state, 'upast')
        [oil_pred, ~] = predict_surrogate_model(model.nlarx_model, u, ...
            state.oil_history(end-model.nlarx_model.na+1:end), ...
            state.upast, model.scaling_params, 0);
    else
        % Simple linear approximation
        % Sensitivity: ~1% oil change per 10 bar BHP change
        bhp_change_bar = BHP - state.BHP;
        oil_pred = state.oil_rate * (1 + 0.001 * bhp_change_bar);
    end
catch
    oil_pred = state.oil_rate;
end

oil_pred = max(oil_pred, 100);  % Minimum feasible production

%% Revenue
revenue = p_oil * oil_pred;

%% Connectivity-Weighted Water Cost
water_cost = 0;
for i = 1:4
    water_cost = water_cost + c_water_base * cost_multipliers(i) * inj(i);
end

%% Control Effort Penalty (BHP in bar)
u_prev = [state.BHP; state.inj_rates(:)];
du = u - u_prev;

% Scale BHP change appropriately (bar has smaller magnitude than psi)
% Weight BHP changes relative to injection changes
% BHP: typically changes by ±10 bar → (±10)² = 100
% Injection: typically changes by ±2000 BBL/d → (±2000)² = 4e6
% To balance, we scale BHP contribution up by ~40x
bhp_scale = 40.0;  % Adjust if needed (20-100 range)

control_penalty = lambda_u * (bhp_scale * du(1)^2 + sum(du(2:5).^2));

%% VRR Penalty (Soft Constraint)
total_inj = sum(inj);
VRR = total_inj / oil_pred;

if VRR > VRR_acceptable
    vrr_penalty = lambda_vrr * (VRR - VRR_acceptable)^2;
elseif VRR < VRR_target * 0.5
    vrr_penalty = 0.1 * lambda_vrr * (VRR_target * 0.5 - VRR)^2;
else
    vrr_penalty = 0;
end

%% BHP Penalty (Pressure Maintenance, bar)
% Penalize deviation from target BHP
% Since BHP is in bar (smaller numbers), scale appropriately
bhp_penalty = lambda_bhp * bhp_scale * (BHP - BHP_target)^2;

%% Total Cost (Minimize = -Profit + Penalties)
J = -revenue + water_cost + control_penalty + vrr_penalty + bhp_penalty;

% Numerical stability
if ~isfinite(J) || isnan(J)
    J = 1e12;
end

end

%% ========== HELPER FUNCTION ==========
function val = get_field(s, field, default)
if isfield(s, field)
    val = s.(field);
else
    val = default;
end
end