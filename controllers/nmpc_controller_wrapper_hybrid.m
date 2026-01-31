function [uo, diagnostics] = nmpc_controller_wrapper_hybrid(yo, upast, nlarx_model, ...
    scaling_params, params, producer_name, connMap)
% NMPC_CONTROLLER_WRAPPER_HYBRID: Production-Grade Wrapper with Economic Tiers
%
% ✅ IMPLEMENTS:
%    - Economic viability pre-screening (Tier 1)
%    - Injector-level economic evaluation (Tier 2)
%    - Adaptive bounds based on economics (Tier 3)
%    - Emergency shut-in logic
%    - Unified initial guess computation
%
% BHP UNITS: bar (native units, no psi conversion)
% Historical baseline: 100 bar (±5 bar)
%
% Based on: Equinor Norne practices + Schlumberger ECLIPSE methods
%
% INPUTS:
%   yo              - Current measured oil rate (STB/d)
%   upast           - Control history [BHP(bar); inj1-4] × time
%   nlarx_model     - Surrogate reservoir model
%   scaling_params  - Model scaling parameters
%   params          - NMPC configuration struct (BHP in bar)
%   producer_name   - Producer well name (e.g., 'P1')
%   connMap         - Connectivity map (optional)
%
% OUTPUTS:
%   uo          - Optimal control [BHP(bar); inj1-4] (5×1)
%   diagnostics - Optimization statistics

%% ========== PERSISTENT VARIABLES ==========
persistent ypred_prev oil_history_buffer bias_history call_count

if isempty(call_count)
    call_count = 0;
    oil_history_buffer = [];
    bias_history = [];
end

call_count = call_count + 1;

%% ========== INPUT VALIDATION ==========
if nargin < 6
    producer_name = 'P1';
    warning('NMPC:NoProducer', 'Producer name not provided. Defaulting to P1.');
end

if nargin < 7 || isempty(connMap)
    connMap = [];
    if call_count == 1
        warning('NMPC:NoConnMap', 'No connectivity map. Using equal weights.');
    end
end

required_fields = {'Np', 'Nu', 'w_oil', 'w_water', 'w_control', ...
                   'BHP_min', 'BHP_max', 'inj_min', 'inj_max'};

for i = 1:length(required_fields)
    if ~isfield(params, required_fields{i})
        error('NMPC:MissingParam', 'params must contain: %s', required_fields{i});
    end
end

if size(upast, 1) ~= 5
    error('NMPC:InvalidDimension', 'upast must be [5 x history_length]');
end

if size(upast, 2) < 1
    error('NMPC:InsufficientHistory', 'upast must have at least 1 sample');
end

BHP_current = upast(1, end);  % bar
inj_current = upast(2:5, end);

verbose = ~isfield(params, 'silent') || ~params.silent;

%% ========== BUILD OIL HISTORY ==========
if isempty(oil_history_buffer)
    oil_history_buffer = yo * ones(max(20, nlarx_model.na), 1);
else
    oil_history_buffer = [oil_history_buffer; yo];
    if length(oil_history_buffer) > 100
        oil_history_buffer = oil_history_buffer(end-99:end);
    end
end

%% ========== TIER 1: ECONOMIC VIABILITY CHECK ==========
if length(oil_history_buffer) >= 10
    recent_oil = mean(oil_history_buffer(end-9:end));
else
    recent_oil = yo;
end

current_total_injection = sum(inj_current);
current_WOR = current_total_injection / (recent_oil + 1e-3);

% Economic parameters
oil_price = params.w_oil;
base_water_cost = get_field(params, 'w_water_base', 5);  % Base cost, not multiplied

% Compute breakeven WOR
WOR_breakeven = oil_price / base_water_cost;  % = 80/5 = 16.0
WOR_economic_limit = 8.0;  % Industry standard

% Check if field is approaching economic limit
if current_WOR > WOR_breakeven * 0.8 && verbose && call_count == 1
    fprintf('\n⚠️  WARNING: Field approaching economic limit!\n');
    fprintf('   Current WOR: %.2f (target: %.1f)\n', current_WOR, WOR_economic_limit);
    fprintf('   Breakeven WOR: %.2f\n', WOR_breakeven);
    fprintf('   → Consider EOR or partial shut-in\n\n');
end

% Emergency shut-in trigger
emergency_shutin = false;
if current_WOR > WOR_economic_limit * 1.5
    emergency_shutin = true;
    if verbose
        fprintf('🚨 EMERGENCY: WOR %.2f >> limit %.1f\n', current_WOR, WOR_economic_limit);
        fprintf('   Forcing aggressive injection reduction...\n\n');
    end
end

%% ========== TIER 2: INJECTOR-LEVEL ECONOMIC SCREENING ==========
if ~isempty(connMap)
    injector_weights = getInjectorWeights(connMap, producer_name);
    injector_weights = injector_weights(1:4);
else
    injector_weights = ones(1, 4);
end

injector_status = zeros(4, 1);  % 1=active, 0=shut-in candidate

for i = 1:4
    % Estimate injector contribution to production
    estimated_oil_response = recent_oil * injector_weights(i) / sum(injector_weights);
    injector_WOR = inj_current(i) / (estimated_oil_response + 1e-3);
    
    % Economic threshold with connectivity adjustment
    well_WOR_limit = WOR_economic_limit * (0.5 + 0.5 * injector_weights(i));
    
    if injector_WOR > well_WOR_limit * 1.2
        injector_status(i) = 0;  % Shut-in candidate
        if verbose && call_count == 1
            fprintf('   ⚠️  I%d: WOR %.1f > limit %.1f (w=%.2f) → SHUT-IN CANDIDATE\n', ...
                i, injector_WOR, well_WOR_limit, injector_weights(i));
        end
    else
        injector_status(i) = 1;  % Keep active
    end
end

%% ========== BIAS CORRECTION ==========
if isempty(ypred_prev)
    bias_correction = 0;
    ypred_prev = yo;
else
    prediction_error = yo - ypred_prev;
    bias_smoothing = 0.3;
    
    if isempty(bias_history)
        bias_correction = bias_smoothing * prediction_error;
    else
        bias_correction = (1 - bias_smoothing) * bias_history(end) + ...
                          bias_smoothing * prediction_error;
    end
    
    bias_history = [bias_history; bias_correction];
    
    if length(bias_history) > 50
        bias_history = bias_history(end-49:end);
    end
end

%% ========== UNIFIED INITIAL GUESS COMPUTATION ==========
if length(oil_history_buffer) >= 10
    oil_decline_rate = (oil_history_buffer(end) - oil_history_buffer(max(1, end-9))) / 10;
else
    oil_decline_rate = 0;
end

% Target oil rate
if isfield(params, 'target_oil_rate') && params.target_oil_rate > 0
    target_oil = params.target_oil_rate;
else
    target_oil = 10000;
end

% BHP target (bar)
if isfield(params, 'BHP_target')
    BHP_target = params.BHP_target;  % bar
else
    BHP_target = (params.BHP_min + params.BHP_max) / 2;  % bar
end

% Connectivity-aware initial guess
if isfield(params, 'u_initial_guess') && ~isempty(params.u_initial_guess)
    initial_guess = params.u_initial_guess;
else
    % Step 1: Determine base fraction based on current conditions
    if oil_decline_rate < -100
        base_fraction = 0.85;  % Aggressive support for steep decline
    elseif recent_oil < 0.7 * target_oil
        base_fraction = 0.75;  % Moderate support for low production
    elseif recent_oil < 0.85 * target_oil
        base_fraction = 0.65;  % Standard operation
    elseif abs(oil_decline_rate) < 20 && abs(recent_oil - target_oil) < 500
        % Steady state - use current control
        initial_guess = [BHP_current; inj_current];
        base_fraction = NaN;  % Signal to skip base allocation
    else
        base_fraction = 0.55;  % Conservative for stable high production
    end
    
    % Step 2: Base allocation with connectivity weighting
    if ~isnan(base_fraction)
        base_rate = base_fraction * params.inj_max;
        
        % Normalize weights to [0.5, 1.5] range
        norm_weights = (injector_weights - min(injector_weights)) / ...
                       (max(injector_weights) - min(injector_weights) + 1e-6);
        connectivity_mult = 0.5 + 1.0 * norm_weights;
        
        % Apply connectivity weighting
        initial_guess_inj = base_rate .* connectivity_mult(:);
        initial_guess_inj = max(params.inj_min, min(initial_guess_inj, params.inj_max));
        initial_guess = [BHP_target; initial_guess_inj];  % BHP in bar
    end
    
    % Step 3: Adaptive scaling for production trends (30-day window)
    if length(oil_history_buffer) >= 60
        recent_30d_avg = mean(oil_history_buffer(end-29:end));
        previous_30d_avg = mean(oil_history_buffer(end-59:end-30));
        production_trend = (recent_30d_avg - previous_30d_avg) / (previous_30d_avg + 1e-3);
        
        if production_trend < -0.15  % >15% decline in last 30 days
            % Reduce injection target to match declining production
            reduction_factor = 0.85;
            initial_guess(2:5) = initial_guess(2:5) * reduction_factor;
            
            if verbose && call_count <= 3
                fprintf('   📉 Production declining %.1f%% → reducing injection by 15%%\n', ...
                    production_trend * 100);
            end
        end
    end
    
    % Step 4: Emergency override for uneconomic wells
    if emergency_shutin
        for i = 1:4
            if injector_status(i) == 0
                initial_guess(i+1) = params.inj_min(i);
                if verbose && call_count <= 3
                    fprintf('   🔴 I%d forced to minimum rate\n', i);
                end
            end
        end
    end
end

%% ========== TIER 3: ADAPTIVE INJECTION BOUNDS ==========
adaptive_inj_min = params.inj_min;
adaptive_inj_max = params.inj_max;

% Global emergency reduction
if emergency_shutin
    if verbose && call_count == 1
        fprintf('   📉 Emergency mode: Reducing injection bounds by 50%%\n');
    end
    adaptive_inj_max = params.inj_min + 0.5 * (params.inj_max - params.inj_min);
end

% Injector-specific bounds based on economics
for i = 1:4
    if injector_status(i) == 0  % Shut-in candidate
        adaptive_inj_max(i) = params.inj_min(i) + 0.3 * (params.inj_max(i) - params.inj_min(i));
        if verbose && call_count <= 3
            fprintf('   🔒 I%d: Max capped at %.0f BBL/d (was %.0f)\n', ...
                i, adaptive_inj_max(i), params.inj_max(i));
        end
    elseif injector_weights(i) > 0.7  % High-quality well
        adaptive_inj_min(i) = max(params.inj_min(i), 8000);
    end
end

%% ========== BUILD STATE STRUCT ==========
current_state = struct();
current_state.BHP = BHP_current;  % bar
current_state.inj_rates = inj_current;
current_state.oil_rate_history = oil_history_buffer;
current_state.upast = upast;
current_state.bias_correction = bias_correction;
current_state.producer_name = producer_name;

%% ========== BUILD CONFIG STRUCT ==========
nmpc_config = struct();
nmpc_config.Np = params.Np;
nmpc_config.Nu = params.Nu;
nmpc_config.use_patternsearch = get_field(params, 'use_patternsearch', false);
nmpc_config.adaptive_tol = get_field(params, 'adaptive_tol', true);
nmpc_config.u_initial_guess = initial_guess;
nmpc_config.silent = isfield(params, 'silent') && params.silent;

% Weights
nmpc_config.weights = struct();
nmpc_config.weights.oil_production = params.w_oil;
nmpc_config.weights.water_injection = get_field(params, 'w_water', 5);
nmpc_config.weights.control_effort = params.w_control;

if isfield(params, 'pressure_weight')
    nmpc_config.weights.pressure_maintenance = params.pressure_weight;
end

% Add connectivity penalty weight
nmpc_config.weights.connectivity_penalty = 2000;  % Tune this (1000-5000 range)

% Constraints (with adaptive bounds) - BHP in bar
nmpc_config.constraints = struct();
nmpc_config.constraints.BHP_min = params.BHP_min;      % bar
nmpc_config.constraints.BHP_max = params.BHP_max;      % bar
nmpc_config.constraints.BHP_target = BHP_target;       % bar
nmpc_config.constraints.inj_min = adaptive_inj_min;    % ✅ Adaptive
nmpc_config.constraints.inj_max = adaptive_inj_max;    % ✅ Adaptive

% Economic context for fallback
nmpc_config.constraints.current_water_cost = get_field(params, 'w_water', 5);
nmpc_config.constraints.current_VRR = current_WOR;
nmpc_config.constraints.dcs_VRR_target = get_field(params, 'VRR_target', 2.5);
nmpc_config.constraints.dcs_VRR_acceptable = get_field(params, 'VRR_acceptable_max', 4.0);

% Additional constraints
nmpc_config.constraints.water_cut_max = get_field(params, 'water_cut_max', 0.92);

if isfield(params, 'target_oil_rate')
    nmpc_config.constraints.target_oil_rate = params.target_oil_rate;
end

if isfield(params, 'min_avg_production')
    nmpc_config.constraints.min_avg_production = params.min_avg_production;
end

if isfield(params, 'max_total_injection')
    nmpc_config.constraints.max_total_injection = params.max_total_injection;
end

% VRR constraint (optional)
if isfield(params, 'enable_VRR_constraint')
    nmpc_config.enable_VRR_constraint = params.enable_VRR_constraint;
    nmpc_config.VRR_target = get_field(params, 'VRR_target', 2.5);
    nmpc_config.VRR_tolerance = get_field(params, 'VRR_tolerance', 1.0);
end

% Store diagnostics
nmpc_config.injector_status = injector_status;
nmpc_config.emergency_mode = emergency_shutin;

%% ========== BUILD MODEL DATA ==========
model_data = struct();
model_data.nlarx_model = nlarx_model;
model_data.scaling_params = scaling_params;

%% ========== CALL NMPC CONTROLLER ==========
try
    [uo, diagnostics] = nmpc_reservoir_hybrid(current_state, nmpc_config, ...
                                              model_data, connMap);
    
    % Update prediction for next bias correction
    if ~isempty(diagnostics.predicted_oil)
        ypred_prev = diagnostics.predicted_oil(1);
    else
        ypred_prev = yo;
    end
    
    % Store injector status in diagnostics
    diagnostics.injector_status = injector_status;
    diagnostics.emergency_mode = emergency_shutin;
    
catch ME
    warning('NMPC:ControllerFailed', 'NMPC failed: %s', ME.message);
    
    % Emergency fallback with connectivity awareness (BHP in bar)
    if oil_decline_rate < -50
        uo = [min(BHP_current + 5.0, params.BHP_max); ...  % +5 bar
              min(inj_current + 2000 * injector_weights(:), adaptive_inj_max)];
    else
        uo = [BHP_current; inj_current];
    end
    
    diagnostics = struct('exitflag', -99, 'iterations', 0, 'funcCount', 0, ...
                         'opt_time', 0, 'final_cost', NaN, ...
                         'predicted_oil', [], 'emergency_fallback', true, ...
                         'error_message', ME.message);
    ypred_prev = yo;
end

%% ========== OUTPUT VALIDATION (BHP in bar) ==========
uo(1) = max(params.BHP_min, min(uo(1), params.BHP_max));  % bar
for i = 1:4
    uo(i+1) = max(adaptive_inj_min(i), min(uo(i+1), adaptive_inj_max(i)));
end

%% ========== PERIODIC SUMMARY (BHP in bar) ==========
if verbose && (mod(call_count, 20) == 0 || call_count <= 3)
    fprintf('\n╔════════════════════════════════════════════════════════╗\n');
    fprintf('║ WRAPPER Call %d Summary\n', call_count);
    fprintf('╠════════════════════════════════════════════════════════╣\n');
    fprintf('║ Oil: %.0f STB/d | WOR: %.2f | Bias: %+.0f STB/d\n', ...
        yo, current_WOR, bias_correction);
    
    if emergency_shutin
        fprintf('║ 🚨 EMERGENCY MODE ACTIVE\n');
    end
    
    fprintf('║ BHP: %.1f → %.1f bar (historical: 100 bar)\n', BHP_current, uo(1));
    fprintf('║ Injection changes:\n');
    
    for i = 1:4
        if injector_status(i) == 0
            status_str = '🔴';
        else
            status_str = '✅';
        end
        fprintf('║   %s I%d: %.0f → %.0f (w=%.2f, Δ%+.0f)\n', ...
            status_str, i, inj_current(i), uo(i+1), injector_weights(i), ...
            uo(i+1) - inj_current(i));
    end
    
    fprintf('╚════════════════════════════════════════════════════════╝\n\n');
end

end

%% ========== HELPER FUNCTIONS ==========
function val = get_field(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end