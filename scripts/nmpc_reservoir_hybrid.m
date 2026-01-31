function [u_opt, diagnostics] = nmpc_reservoir_hybrid(current_state, nmpc_config, model_data, connMap)
% NMPC_RESERVOIR_HYBRID - Hierarchical economic NMPC with optimization-based fallback
%
% ARCHITECTURE:
%   Layer 1 (Primary): Full nonlinear MPC - long horizon, detailed model
%   Layer 2 (Fallback): Reduced-order economic MPC - single-step, fast solve
%   Both layers solve formal optimization problems
%
% BHP UNITS: bar (native units, no psi conversion)
% Historical baseline: 100 bar (±5 bar)
%
% LAYER 1 FORMULATION:
%   minimize: J = Σ[-p_oil*q_oil + c_water*q_water + λ_u||Δu||² + penalties]
%   s.t.: dynamics, bounds (BHP in bar), VRR constraints (optional)
%   solver: fmincon (SQP) or patternsearch
%
% LAYER 2 FORMULATION (when Layer 1 fails):
%   maximize: J = p_oil*q_oil - c_water(VRR)*q_water - λ||Δu||²
%   s.t.: bounds (BHP in bar), soft VRR constraint
%   solver: fminsearch (Nelder-Mead)
%
% INPUTS:
%   current_state - struct(BHP(bar), inj_rates, oil_rate_history, upast, 
%                          bias_correction, producer_name)
%   nmpc_config   - struct(Np, Nu, weights, constraints(BHP in bar), u_initial_guess, ...)
%   model_data    - struct(nlarx_model, scaling_params)
%   connMap       - connectivity map (optional)
%
% OUTPUTS:
%   u_opt       - optimal control [BHP(bar); inj1-4] (5×1)
%   diagnostics - optimization statistics

persistent u_prev steady_state_counter call_count u_prev_solution

if isempty(call_count)
    call_count = 0;
    steady_state_counter = 0;
    u_prev_solution = [];
end
call_count = call_count + 1;

%% ========== EXTRACT INPUTS ==========
BHP_current = current_state.BHP;  % bar
inj_current = current_state.inj_rates(:);
oil_history = current_state.oil_rate_history(:);
upast = current_state.upast;
bias_correction = current_state.bias_correction;
producer_name = current_state.producer_name;

Np = nmpc_config.Np;
Nu = nmpc_config.Nu;
weights = nmpc_config.weights;
constraints = nmpc_config.constraints;
nlarx_model = model_data.nlarx_model;
scaling_params = model_data.scaling_params;

%% ========== GET INJECTOR CONNECTIVITY WEIGHTS ==========
if nargin < 4 || isempty(connMap)
    injector_weights = ones(1, 4);
else
    injector_weights = getInjectorWeights(connMap, producer_name);
    injector_weights = injector_weights(1:4);
end

%% ========== VALIDATE INPUTS ==========
na = nlarx_model.na;
if length(inj_current) ~= 4
    error('NMPC:InvalidInput', 'inj_rates must be [4x1]');
end

if length(oil_history) < na
    error('NMPC:InsufficientHistory', 'Need %d samples, got %d', na, length(oil_history));
end

% Ensure oil history is exactly na samples
oil_history_trimmed = oil_history(max(1, end-na+1):end);
if length(oil_history_trimmed) < na
    oil_history_trimmed = [repmat(oil_history(1), na-length(oil_history_trimmed), 1); 
                           oil_history_trimmed];
end

verbose = ~isfield(nmpc_config, 'silent') || ~nmpc_config.silent;

%% ========== EXTRACT ECONOMIC CONTEXT (for fallback) ==========
current_water_cost = get_field(constraints, 'current_water_cost', 7.5);
current_VRR = get_field(constraints, 'current_VRR', ...
    sum(inj_current) / (oil_history(end) + 1e-3));
dcs_VRR_target = get_field(constraints, 'dcs_VRR_target', 2.5);
dcs_VRR_acceptable = get_field(constraints, 'dcs_VRR_acceptable', 4.0);

%% ========== STEADY-STATE DETECTION ==========
if ~isempty(u_prev)
    u_current = [BHP_current; inj_current];  % BHP in bar
    du_norm = abs(u_current - u_prev) ./ (abs(u_prev) + 1e-6);
    
    % Check for production decline
    if length(oil_history) >= 10
        decline_rate = (oil_history(end) - oil_history(end-9)) / 10;
        declining_fast = decline_rate < -50;
    else
        declining_fast = false;
    end
    
    % Increment steady-state counter
    if max(du_norm) < 5e-3 && ~declining_fast
        steady_state_counter = steady_state_counter + 1;
    else
        steady_state_counter = 0;
    end
    
    % Check if at boundary (BHP in bar)
    at_boundary = abs(BHP_current - constraints.BHP_min) < 2.0 || ...  % 2 bar tolerance
                  abs(BHP_current - constraints.BHP_max) < 2.0 || ...
                  any(abs(inj_current - constraints.inj_min) < 100) || ...
                  any(abs(inj_current - constraints.inj_max) < 100);
    
    % Skip optimization if truly steady and at boundary
    skip_opt = (steady_state_counter >= 5) && at_boundary && ~declining_fast;
    
    % Override skip for critical VRR violations
    if current_VRR > 9.0
        skip_opt = false;
    end
    
    if skip_opt
        if verbose && call_count <= 3
            fprintf('[NMPC] Steady state detected - skipping optimization\n');
        end
        u_opt = u_current;
        diagnostics = struct('exitflag', 1, 'iterations', 0, 'funcCount', 0, ...
            'opt_time', 0, 'final_cost', 0, ...
            'predicted_oil', oil_history(end) * ones(Np, 1), ...
            'steady_state_skip', true);
        return;
    end
end
u_prev = [BHP_current; inj_current];  % BHP in bar

%% ========== WARM START (CONNECTIVITY-AWARE) ==========
if ~isempty(u_prev_solution) && length(u_prev_solution) == 5*Nu
    u0_physical = u_prev_solution;
elseif isfield(nmpc_config, 'u_initial_guess') && ~isempty(nmpc_config.u_initial_guess)
    u0_physical = repmat(nmpc_config.u_initial_guess, Nu, 1);
else
    % Compute connectivity-weighted initial guess
    % Start with connectivity-proportional allocation
    injector_weights_norm = injector_weights / sum(injector_weights);
    
    % Target total injection (from constraints or current state)
    if isfield(constraints, 'max_total_injection')
        target_total = 0.7 * constraints.max_total_injection;  % Start at 70%
    else
        target_total = sum(inj_current);
    end
    
    % Allocate proportionally to connectivity
    inj_guess = zeros(4, 1);
    for i = 1:4
        inj_guess(i) = injector_weights_norm(i) * target_total;
        
        % Ensure within bounds
        inj_guess(i) = max(constraints.inj_min(i), ...
                           min(inj_guess(i), constraints.inj_max(i)));
    end
    
    % Adjust BHP guess based on injection level (bar)
    BHP_guess = BHP_current;
    if target_total > sum(inj_current)
        BHP_guess = min(BHP_current + 5.0, constraints.BHP_max);  % +5 bar
    end
    
    u0_single = [BHP_guess; inj_guess];  % BHP in bar
    u0_physical = repmat(u0_single, Nu, 1);
end

% Verify warm start feasibility (BHP in bar)
u0_seq = reshape(u0_physical, 5, Nu);
for k = 1:Nu
    % Project onto bounds (BHP in bar)
    u0_seq(1, k) = max(constraints.BHP_min, min(u0_seq(1, k), constraints.BHP_max));
    for i = 1:4
        u0_seq(i+1, k) = max(constraints.inj_min(i), ...
                             min(u0_seq(i+1, k), constraints.inj_max(i)));
    end
    
    % Check total injection constraint
    total_inj_k = sum(u0_seq(2:5, k));
    if isfield(constraints, 'max_total_injection') && ...
       total_inj_k > constraints.max_total_injection
        scale = 0.95 * constraints.max_total_injection / total_inj_k;
        u0_seq(2:5, k) = u0_seq(2:5, k) * scale;
    end
end
u0_physical = u0_seq(:);

%% ========== BOUNDS (BHP in bar) ==========
lb = repmat([constraints.BHP_min; constraints.inj_min(:)], Nu, 1);
ub = repmat([constraints.BHP_max; constraints.inj_max(:)], Nu, 1);

%% ========== RATE-OF-CHANGE CONSTRAINTS (BHP in bar) ==========
A_ineq = [];
b_ineq = [];

for k = 1:Nu-1
    for i = 1:5
        % Max rate of change: 10 bar/day for BHP, 2000 BBL/day for injection
        max_delta = iif(i==1, 10.0, 2000);  % bar or BBL/day
        
        % Δu[k] ≤ max_delta
        A_row = zeros(1, 5*Nu);
        A_row(5*(k-1)+i) = -1;
        A_row(5*k+i) = 1;
        A_ineq = [A_ineq; A_row];
        b_ineq = [b_ineq; max_delta];
        
        % -Δu[k] ≤ max_delta (i.e., Δu[k] ≥ -max_delta)
        A_row = zeros(1, 5*Nu);
        A_row(5*(k-1)+i) = 1;
        A_row(5*k+i) = -1;
        A_ineq = [A_ineq; A_row];
        b_ineq = [b_ineq; max_delta];
    end
end

%% ========== TOTAL INJECTION CONSTRAINT ==========
if isfield(constraints, 'max_total_injection') && constraints.max_total_injection > 0
    for k = 1:Nu
        A_row = zeros(1, 5*Nu);
        A_row(5*(k-1)+2:5*(k-1)+5) = 1;
        A_ineq = [A_ineq; A_row];
        b_ineq = [b_ineq; constraints.max_total_injection];
    end
end

%% ========== COST FUNCTION ==========
% DIAGNOSTIC: Verify injector weights are being passed
if call_count == 1 && verbose
    fprintf('\n[DEBUG] Cost function setup:\n');
    fprintf('  Injector weights: [%.3f, %.3f, %.3f, %.3f]\n', injector_weights);
    fprintf('  Oil price: $%.0f/STB\n', weights.oil_production);
    fprintf('  Base water cost: $%.2f/BBL\n', weights.water_injection);
    fprintf('  BHP bounds: %.1f - %.1f bar\n', constraints.BHP_min, constraints.BHP_max);
end

cost_fun = @(u_vec) nmpc_cost_function_hybrid(u_vec, Np, Nu, ...
    oil_history_trimmed, upast, nlarx_model, scaling_params, ...
    weights, constraints, bias_correction, injector_weights);

%% ========== NONLINEAR CONSTRAINTS ==========
nonlcon = [];

% Production constraint
if isfield(constraints, 'min_avg_production') && constraints.min_avg_production > 0
    nonlcon = @(u) production_constraint(u, Np, Nu, oil_history_trimmed, upast, ...
        nlarx_model, scaling_params, constraints.min_avg_production, bias_correction);
end

% VRR constraint (optional)
if isfield(nmpc_config, 'enable_VRR_constraint') && nmpc_config.enable_VRR_constraint
    nonlcon_vrr = @(u) vrr_constraint(u, Np, Nu, oil_history_trimmed, upast, ...
        nlarx_model, scaling_params, nmpc_config.VRR_target, ...
        nmpc_config.VRR_tolerance, bias_correction);
    
    if isempty(nonlcon)
        nonlcon = nonlcon_vrr;
    else
        nonlcon_old = nonlcon;
        nonlcon = @(u) combine_constraints(u, nonlcon_old, nonlcon_vrr);
    end
end

%% ========== OPTIMIZER SELECTION & OPTIONS ==========
use_ps = isfield(nmpc_config, 'use_patternsearch') && nmpc_config.use_patternsearch;

% Adaptive tolerance based on configuration
if isfield(nmpc_config, 'adaptive_tol') && nmpc_config.adaptive_tol
    opt_tol = 5e-2;
    max_iter = 80;
    max_feval = 3000;
else
    opt_tol = 1e-3;
    max_iter = 120;
    max_feval = 4000;
end

%% ========== RUN PRIMARY OPTIMIZATION (LAYER 1) ==========
tic;
if use_ps
    % Pattern Search
    ps_opts = optimoptions('patternsearch', 'Display', 'off', ...
        'MaxIterations', 100*(5*Nu), 'MaxFunctionEvaluations', max_feval, ...
        'MeshTolerance', opt_tol, 'StepTolerance', opt_tol, ...
        'ConstraintTolerance', 1e-4, 'UseCompletePoll', true, ...
        'UseCompleteSearch', false);
    
    if ~isempty(nonlcon)
        % Add penalty for nonlinear constraints
        cost_pen = @(u) cost_fun(u) + compute_penalty(u, nonlcon);
        [u_opt_vec, fval, exitflag, output] = patternsearch(cost_pen, u0_physical, ...
            A_ineq, b_ineq, [], [], lb, ub, [], ps_opts);
    else
        [u_opt_vec, fval, exitflag, output] = patternsearch(cost_fun, u0_physical, ...
            A_ineq, b_ineq, [], [], lb, ub, [], ps_opts);
    end
else
    % fmincon (SQP)
    fmincon_opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
        'MaxIterations', max_iter, 'MaxFunctionEvaluations', max_feval*2, ...
        'OptimalityTolerance', opt_tol, 'StepTolerance', opt_tol*10, ...
        'ConstraintTolerance', 1e-5, 'ScaleProblem', true, ...
        'FiniteDifferenceType', 'central');
    
    [u_opt_vec, fval, exitflag, output] = fmincon(cost_fun, u0_physical, ...
        A_ineq, b_ineq, [], [], lb, ub, nonlcon, fmincon_opts);
end
opt_time = toc;

% Store successful solution for warm start
if exitflag >= 0
    u_prev_solution = u_opt_vec;
end

% Extract first control from sequence (BHP in bar)
u_opt_seq = reshape(u_opt_vec, 5, Nu);
u_opt = u_opt_seq(:, 1);

%% ========== LAYER 2: ECONOMIC MPC FALLBACK ==========
if exitflag < 0
    if verbose && call_count <= 5
        fprintf('[NMPC Failed exitflag=%d] → Economic MPC fallback\n', exitflag);
    end
    
    % Build economic MPC state (BHP in bar)
    econ_state = struct('BHP', BHP_current, 'inj_rates', inj_current, ...
        'oil_rate', oil_history(end), 'injector_weights', injector_weights, ...
        'oil_history', oil_history, 'upast', upast);
    
    econ_constraints = struct('inj_min', constraints.inj_min, ...
        'inj_max', constraints.inj_max, 'BHP_min', constraints.BHP_min, ...
        'BHP_max', constraints.BHP_max, 'VRR_target', dcs_VRR_target, ...
        'VRR_acceptable_max', dcs_VRR_acceptable);
    
    if isfield(constraints, 'max_total_injection')
        econ_constraints.max_total_injection = constraints.max_total_injection;
    end
    
    econ_model = struct('nlarx_model', nlarx_model, 'scaling_params', scaling_params);
    
    % Economic configuration
    econ_config = struct('oil_price', 80, 'base_water_cost', 5, ...
        'control_weight', 100, 'VRR_penalty', 5e4, 'BHP_penalty', 1e3);
    
    if exist('weights', 'var')
        econ_config.oil_price = get_field(weights, 'oil_production', 80);
        econ_config.base_water_cost = get_field(weights, 'water_injection', 5);
        econ_config.control_weight = get_field(weights, 'control_effort', 100);
    end
    
    try
        [u_opt, econ_diag] = economic_mpc_fallback(econ_state, econ_constraints, ...
            econ_model, econ_config);
        
        diagnostics = struct('exitflag', econ_diag.exitflag, ...
            'iterations', econ_diag.iterations, 'funcCount', econ_diag.funcCount, ...
            'opt_time', econ_diag.opt_time, 'final_cost', econ_diag.objective_value, ...
            'predicted_oil', econ_diag.oil_predicted, 'fallback_used', true, ...
            'fallback_type', 'ECONOMIC_MPC', 'VRR_predicted', econ_diag.VRR_predicted, ...
            'water_cost', econ_diag.water_cost, 'primary_exitflag', exitflag);
        
        if verbose && mod(call_count, 20) == 0
            fprintf('[EconMPC] Day %d: Oil=%.0f | VRR=%.2f | Cost=$%.2f/BBL\n', ...
                call_count, econ_diag.oil_predicted, econ_diag.VRR_predicted, econ_diag.water_cost);
        end
    catch ME
        warning('EconomicMPC:Failed', 'Fallback failed: %s', ME.message);
        u_opt = [BHP_current; inj_current];  % BHP in bar
        diagnostics = struct('exitflag', -99, 'iterations', 0, 'funcCount', 0, ...
            'opt_time', 0, 'final_cost', NaN, 'predicted_oil', [], ...
            'fallback_used', true, 'fallback_type', 'HOLD_LAST', ...
            'error_message', ME.message, 'primary_exitflag', exitflag);
    end
    
    % Enforce bounds on fallback solution (BHP in bar)
    u_opt(1) = max(constraints.BHP_min, min(u_opt(1), constraints.BHP_max));
    for i = 1:4
        u_opt(i+1) = max(constraints.inj_min(i), min(u_opt(i+1), constraints.inj_max(i)));
    end
else
    %% ========== PRIMARY NMPC SUCCEEDED ==========
    diagnostics = struct('exitflag', exitflag, 'iterations', output.iterations, ...
        'funcCount', get_field(output, 'funcCount', get_field(output, 'funccount', 0)), ...
        'opt_time', opt_time, 'final_cost', fval, 'u_opt_sequence', u_opt_seq, ...
        'fallback_used', false, 'steady_state_skip', false, ...
        'injector_weights', injector_weights, 'warm_start_used', ~isempty(u_prev_solution), ...
        'optimizer_used', iif(use_ps, 'patternsearch', 'fmincon'));
    
    % Predict oil with optimal control
    u_full_opt = [u_opt_seq, repmat(u_opt_seq(:,end), 1, Np-Nu)];
    try
        [y_pred, ~] = predict_surrogate_model(nlarx_model, u_full_opt, ...
            oil_history_trimmed, upast, scaling_params, 0);
        diagnostics.predicted_oil = y_pred + bias_correction;
        
        % Calculate predicted VRR
        total_inj_pred = sum(u_full_opt(2:5, :), 'all');
        total_oil_pred = sum(y_pred + bias_correction);
        diagnostics.VRR_predicted = total_inj_pred / (total_oil_pred + 1e-3);
    catch ME
        diagnostics.predicted_oil = [];
        diagnostics.VRR_predicted = current_VRR;
        if verbose
            warning('NMPC:PredictionFailed', 'Could not predict oil: %s', ME.message);
        end
    end
end

end

%% ========== HELPER FUNCTIONS ==========

function [c, ceq] = production_constraint(u_vec, Np, Nu, ypast, upast, ...
    model, scale, min_prod, bias)
% Ensure average production is above minimum
u_seq = reshape(u_vec, 5, Nu);
u_full = [u_seq, repmat(u_seq(:,end), 1, Np-Nu)];
try
    [y, ~] = predict_surrogate_model(model, u_full, ypast, upast, scale, 0);
    c = min_prod - mean(y + bias);  % c ≤ 0 for feasibility
catch
    c = 1e6;  % Infeasible if prediction fails
end
ceq = [];
end

function [c, ceq] = vrr_constraint(u_vec, Np, Nu, ypast, upast, ...
    model, scale, VRR_tgt, VRR_tol, bias)
% Keep VRR within target ± tolerance
u_seq = reshape(u_vec, 5, Nu);
u_full = [u_seq, repmat(u_seq(:,end), 1, Np-Nu)];
try
    [y, ~] = predict_surrogate_model(model, u_full, ypast, upast, scale, 0);
    y = y + bias;
    VRR = sum(u_full(2:5,:), 'all') / (sum(y) + 1e-3);
    
    % Two-sided constraint: VRR_target - tol ≤ VRR ≤ VRR_target + tol
    c = [(VRR_tgt - VRR_tol) - VRR;   % Lower bound
         VRR - (VRR_tgt + VRR_tol)];  % Upper bound
catch
    c = [1e6; 1e6];  % Infeasible if prediction fails
end
ceq = [];
end

function [c, ceq] = combine_constraints(u, con1, con2)
% Combine two constraint functions
[c1, ceq1] = con1(u);
[c2, ceq2] = con2(u);
c = [c1; c2];
ceq = [ceq1; ceq2];
end

function pen = compute_penalty(u, nonlcon)
% Compute penalty for nonlinear constraint violations
try
    [c, ceq] = nonlcon(u);
    pen = 1e6 * (sum(max(0, c)) + sum(abs(ceq)));
catch
    pen = 1e8;
end
end

function val = get_field(s, field, default)
% Safe field extraction with default
if isfield(s, field)
    val = s.(field);
else
    val = default;
end
end

function result = iif(cond, true_val, false_val)
% Inline if-else
if cond
    result = true_val;
else
    result = false_val;
end
end