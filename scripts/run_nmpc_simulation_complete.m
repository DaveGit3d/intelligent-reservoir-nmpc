%% ============================================================================
%% run_nmpc_simulation.m - Main simulation without plotting
%% Historical baseline: 100 bar (±5 bar) constant BHP regulation
%% ============================================================================

clear; clc; close all;

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  HIERARCHICAL ECONOMIC NMPC                    ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% ========== BUILD CONNECTIVITY MAP ==========
connMap = buildConnectivityMap();

%% ========== CONFIGURATION ==========
config = struct();
config.N_steps = 365;
config.Ts = 1.0;
config.wellName = 'P1';
config.verbose = true;
config.enable_plots = false;  % DISABLED
config.run_baseline = false;
config.dcs_update_interval = 10;
config.dcs_mode = 'AGGRESSIVE';
config.oil_price = 80;
config.base_water_cost = 5;
config.opex_daily = 50000;
config.annual_discount_rate = 0.10;
config.water_cost_max = 20.0;

%% ========== LOAD DATA ==========
P1_data = readtable(sprintf('%s.csv', config.wellName));

if exist('P1_NLARX_NMPC_updated.mat', 'file')
    load('P1_NLARX_NMPC_updated.mat', 'sys_new', 'scaling_params');
    nlarx_model = sys_new;
else
    load('P1_NLARX_NMPC.mat', 'nlarx_model', 'scaling_params');
end

%% ========== GET INJECTOR WEIGHTS ==========
weights_P1 = getInjectorWeights(connMap, config.wellName);
weights_P1 = weights_P1(1:4);

%% ========== BASE CONSTRAINTS ==========
% Historical baseline: 100 bar ± 5 bar constant BHP regulation
% NMPC constraints: Allow wider range for optimization
base_constraints = struct();
base_constraints.BHP_min = 80;           % bar (20% below baseline for flexibility)
base_constraints.BHP_max = 150;          % bar (50% above baseline for control)
base_constraints.BHP_target = 100;       % bar (matches historical baseline)
base_constraints.inj_min = 5000 * ones(4, 1);
base_constraints.inj_max = 30000 * ones(4, 1);
base_constraints.max_total_injection = 80000;
base_constraints.water_cut_max = 0.95;

fprintf('BHP constraints (bar, native units):\n');
fprintf('  Min:    %.0f bar (allows drawdown for stimulation)\n', base_constraints.BHP_min);
fprintf('  Max:    %.0f bar (prevents excessive pressure)\n', base_constraints.BHP_max);
fprintf('  Target: %.0f bar (historical baseline)\n', base_constraints.BHP_target);
fprintf('  Historical operation: 95-105 bar (±5 bar)\n\n');

%% ========== NMPC PARAMETERS ==========
params = struct();
params.Np = 8;
params.Nu = 4;
params.w_oil = config.oil_price;
params.w_water = config.base_water_cost;
params.w_control = 100;
params.use_patternsearch = false;
params.adaptive_tol = true;

%% ========== INITIALIZE STATE VARIABLES ==========
time = zeros(config.N_steps, 1);
oil_measured = zeros(config.N_steps, 1);
oil_setpoint = zeros(config.N_steps, 1);
predicted_oil = zeros(config.N_steps, 1);
BHP_actual = zeros(config.N_steps, 1);      
BHP_command = zeros(config.N_steps, 1);     
inj_actual = zeros(config.N_steps, 4);
inj_command = zeros(config.N_steps, 4);
opt_time = zeros(config.N_steps, 1);
status_history = zeros(config.N_steps, 1);

WOR_history = zeros(config.N_steps, 1);
VRR_history = zeros(config.N_steps, 1);
water_cut_history = zeros(config.N_steps, 1);
vrr_multiplier_history = zeros(config.N_steps, 1);
marginal_cost_ratio_history = zeros(config.N_steps, 1);

daily_oil_revenue = zeros(config.N_steps, 1);
daily_water_cost = zeros(config.N_steps, 1);
daily_npv = zeros(config.N_steps, 1);
cumulative_npv = zeros(config.N_steps, 1);
effective_water_cost_per_bbl = zeros(config.N_steps, 1);

dcs_active_injectors = zeros(config.N_steps, 4);
dcs_max_total_injection = zeros(config.N_steps, 1);
dcs_water_multiplier = zeros(config.N_steps, 1);
dcs_policy_history = cell(config.N_steps, 1);
dcs_vrr_target_history = zeros(config.N_steps, 1);
dcs_vrr_acceptable_history = zeros(config.N_steps, 1);

fallback_used_history = zeros(config.N_steps, 1);
fallback_reason_history = cell(config.N_steps, 1);

water_cost_capped_count = 0;
water_cost_uncapped_total = 0;
water_cost_savings_total = 0;

%% ========== INITIALIZE ACTUATORS (BHP IN BAR) ==========
actuators = struct();
actuators.BHP_actual = base_constraints.BHP_target;     
actuators.BHP_command = base_constraints.BHP_target;    
actuators.BHP_alpha = 0.7;

total_initial_inj = 60000;
initial_allocation = total_initial_inj * (weights_P1 / sum(weights_P1));
actuators.inj_actual = initial_allocation(:);
actuators.inj_command = initial_allocation(:);
actuators.inj_alpha = 0.7;

fprintf('Initial actuator state:\n');
fprintf('  BHP: %.1f bar (historical baseline)\n', actuators.BHP_actual);
fprintf('  Total injection: %.0f BBL/d\n', sum(actuators.inj_actual));

%% ========== INITIALIZE DCS ==========
dcs = struct();
dcs.oil_baseline = 10000;
dcs.oil_current = dcs.oil_baseline;
dcs.oil_setpoint = dcs.oil_baseline;
dcs.alpha = 0.8;
dcs.noise_level = 100;
dcs_policy = [];

%% ========== INITIALIZE HISTORY BUFFERS ==========
ypast_current = dcs.oil_baseline * ones(max(20, nlarx_model.na), 1);
u_initial = [actuators.BHP_actual; actuators.inj_actual];  
nb_max = max(nlarx_model.nb(:));
upast_current = repmat(u_initial, 1, max(10, nb_max));
uo_current = u_initial;

fprintf('\n✓ Initialization complete. Starting simulation...\n\n');

%% ========== MAIN SIMULATION LOOP ==========
clear nmpc_controller_wrapper_hybrid;
sim_start_time = tic;

cumulative_oil = 0;
cumulative_water = 0;
cumulative_revenue_discounted = 0;
cumulative_water_cost = 0;
cumulative_opex = 0;
discount_factor_daily = exp(-config.annual_discount_rate * config.Ts / 365);

economic_limit_triggered = false;
economic_limit_day = -1;
nmpc_success_count = 0;
nmpc_total_count = 0;
fallback_count = 0;
dcs_update_counter = 0;
water_cost_cap_warning_shown = false;

for k = 1:config.N_steps
    time(k) = (k-1) * config.Ts;
    
    %% Field State Tracking
    if k > 1
        lookback = min(k-1, 30);
        recent_water_cost = mean(effective_water_cost_per_bbl(max(1,k-lookback):k-1));
        recent_oil_revenue = mean(oil_measured(max(1,k-lookback):k-1)) * config.oil_price;
        marginal_cost_ratio_current = (recent_water_cost * mean(sum(inj_actual(max(1,k-lookback):k-1,:), 2))) / ...
            (recent_oil_revenue + 1e-6);
        marginal_cost_ratio_history(k) = marginal_cost_ratio_current;
    else
        marginal_cost_ratio_current = 0;
        marginal_cost_ratio_history(k) = 0;
    end
    
    %% DCS Supervisor
    if mod(k-1, config.dcs_update_interval) == 0 || k == 1
        dcs_update_counter = dcs_update_counter + 1;
        field_state = struct();
        
        if k == 1
            field_state.WOR = sum(actuators.inj_actual) / dcs.oil_baseline;
            field_state.water_cut = field_state.WOR / (1 + field_state.WOR);
            field_state.avg_oil = dcs.oil_baseline;
            field_state.avg_water = sum(actuators.inj_actual);
            field_state.oil_trend = 0;
            field_state.marginal_cost_ratio = 0;
        else
            lookback = min(k-1, config.dcs_update_interval);
            field_state.avg_oil = mean(oil_measured(max(1,k-lookback):k-1));
            field_state.avg_water = mean(sum(inj_actual(max(1,k-lookback):k-1,:), 2));
            field_state.WOR = field_state.avg_water / (field_state.avg_oil + 1e-3);
            field_state.water_cut = field_state.WOR / (1 + field_state.WOR);
            
            if k > config.dcs_update_interval
                recent_avg = mean(oil_measured(k-config.dcs_update_interval:k-1));
                previous_avg = mean(oil_measured(max(1,k-2*config.dcs_update_interval):max(1,k-config.dcs_update_interval-1)));
                field_state.oil_trend = (recent_avg - previous_avg) / config.dcs_update_interval;
            else
                field_state.oil_trend = 0;
            end
            
            field_state.marginal_cost_ratio = marginal_cost_ratio_current;
        end
        
        field_state.injector_weights = weights_P1(1:4);
        field_state.current_day = k;
        field_state.avg_water_cost = config.base_water_cost * ...
            (1.0 + 0.5 * field_state.water_cut + 0.1 * max(0, field_state.WOR - 3.0));
        
        verbose_dcs = config.verbose && (dcs_update_counter <= 3 || mod(dcs_update_counter, 5) == 0);
        dcs_policy = dcs_supervisor(field_state, dcs_policy, base_constraints, verbose_dcs);
        
        if economic_limit_triggered
            dcs_policy.max_total_injection = 12000;
            dcs_policy.water_cost_multiplier = 5.0;
            dcs_policy.BHP_target = base_constraints.BHP_min + 5.0;  % 85 bar in survival mode
        end
    end
    
    %% Actuator Dynamics (BHP in bar)
    actuators.BHP_actual = actuators.BHP_alpha * actuators.BHP_actual + ...
        (1 - actuators.BHP_alpha) * actuators.BHP_command;
    actuators.inj_actual = actuators.inj_alpha * actuators.inj_actual + ...
        (1 - actuators.inj_alpha) * actuators.inj_command;
    
    %% Reservoir Response
    try
        u_actual = [actuators.BHP_actual; actuators.inj_actual];  
        [oil_pred, ~] = predict_surrogate_model(nlarx_model, u_actual, ...
            ypast_current(end-nlarx_model.na+1:end), upast_current, scaling_params, 0);
        dcs.oil_setpoint = oil_pred(1);
    catch ME
        if config.verbose && k <= 3
            warning(['Prediction failed: ' ME.message]);
        end
        dcs.oil_setpoint = dcs.oil_current - 50;
    end
    
    dcs.oil_current = dcs.alpha * dcs.oil_current + (1 - dcs.alpha) * dcs.oil_setpoint;
    oil_noisy = dcs.oil_current + randn() * dcs.noise_level;
    oil_noisy = max(1000, min(50000, oil_noisy));
    
    oil_measured(k) = oil_noisy;
    oil_setpoint(k) = dcs.oil_setpoint;
    BHP_actual(k) = actuators.BHP_actual;  
    inj_actual(k, :) = actuators.inj_actual';
    
    %% Daily Economics with Water Cost Cap
    discount_k = discount_factor_daily ^ (k-1);
    daily_oil_revenue(k) = oil_noisy * config.oil_price * discount_k;
    cumulative_revenue_discounted = cumulative_revenue_discounted + daily_oil_revenue(k);
    
    oil_k = max(oil_noisy, 1e-3);
    water_k = sum(actuators.inj_actual);
    VRR_k = water_k / oil_k;
    
    if VRR_k < 4.0
        vrr_mult = 0.82 * exp(0.12 * VRR_k);
    else
        vrr_mult = 1.5 * exp(0.35 * (VRR_k - 4.0));
    end
    vrr_mult = min(vrr_mult, 4.0);
    vrr_multiplier_history(k) = vrr_mult;
    
    total_water_multiplier = vrr_mult * dcs_policy.water_cost_multiplier;
    c_water_max = config.water_cost_max;
    uncapped_water_cost = config.base_water_cost * total_water_multiplier;
    effective_water_cost_per_bbl(k) = min(uncapped_water_cost, c_water_max);
    
    if uncapped_water_cost > c_water_max
        water_cost_capped_count = water_cost_capped_count + 1;
        water_cost_uncapped_total = water_cost_uncapped_total + uncapped_water_cost;
        water_cost_savings_total = water_cost_savings_total + (uncapped_water_cost - c_water_max);
    end
    
    daily_water_cost(k) = water_k * effective_water_cost_per_bbl(k) * discount_k;
    cumulative_water_cost = cumulative_water_cost + daily_water_cost(k);
    
    daily_opex = config.opex_daily * discount_k;
    cumulative_opex = cumulative_opex + daily_opex;
    
    daily_npv(k) = daily_oil_revenue(k) - daily_water_cost(k) - daily_opex;
    cumulative_npv(k) = cumulative_revenue_discounted - cumulative_water_cost - cumulative_opex;
    
    cumulative_oil = cumulative_oil + oil_noisy * config.Ts;
    cumulative_water = cumulative_water + water_k * config.Ts;
    
    %% Economic Limit Check
    if k > 30 && ~economic_limit_triggered
        avg_daily_npv = mean(daily_npv(k-29:k));
        water_to_oil_ratio = mean(effective_water_cost_per_bbl(k-29:k)) * ...
            mean(sum(inj_actual(k-29:k,:), 2)) / ...
            (mean(oil_measured(k-29:k)) * config.oil_price + 1e-6);
        
        if avg_daily_npv < -50000 && (water_to_oil_ratio > 2.0 || mean(VRR_history(k-29:k)) > 7.0)
            economic_limit_triggered = true;
            economic_limit_day = k;
            fprintf('\n🛑 ECONOMIC LIMIT REACHED AT DAY %d\n\n', k);
        end
    end
    
    %% VRR and Water Cut
    if oil_noisy > 100
        WOR_history(k) = water_k / oil_noisy;
        VRR_history(k) = water_k / oil_noisy;
    else
        WOR_history(k) = 20;
        VRR_history(k) = 20;
    end
    water_cut_history(k) = WOR_history(k) / (1 + WOR_history(k));
    
    ypast_current = [ypast_current(2:end); oil_noisy];
    
    %% Store DCS State
    dcs_active_injectors(k, :) = dcs_policy.inj_enabled';
    dcs_max_total_injection(k) = dcs_policy.max_total_injection;
    dcs_water_multiplier(k) = dcs_policy.water_cost_multiplier;
    dcs_policy_history{k} = dcs_policy.policy_reason;
    dcs_vrr_target_history(k) = dcs_policy.VRR_target;
    dcs_vrr_acceptable_history(k) = dcs_policy.VRR_acceptable_max;
    
    %% NMPC Controller (BHP in bar)
    params.BHP_target = dcs_policy.BHP_target;      
    params.BHP_min = base_constraints.BHP_min;      
    params.BHP_max = base_constraints.BHP_max;      
    params.inj_min = dcs_policy.inj_min;
    params.inj_max = dcs_policy.inj_max;
    params.max_total_injection = dcs_policy.max_total_injection;
    params.water_cut_max = base_constraints.water_cut_max;
    params.w_water = config.base_water_cost * dcs_policy.water_cost_multiplier;
    params.current_water_cost = effective_water_cost_per_bbl(max(1,k-1));
    params.current_VRR = water_k / (oil_k + 1e-3);
    params.dcs_VRR_target = dcs_policy.VRR_target;
    params.dcs_VRR_acceptable = dcs_policy.VRR_acceptable_max;
    params.use_patternsearch = (k <= 10);
    params.adaptive_tol = (k > 5);
    params.silent = k > 10;
    
    nmpc_total_count = nmpc_total_count + 1;
    t_opt_start = tic;
    
    try
        [u_opt, diagnostics] = nmpc_controller_wrapper_hybrid(oil_noisy, upast_current, ...
            nlarx_model, scaling_params, params, config.wellName, connMap);
        opt_time(k) = toc(t_opt_start);
        status_history(k) = diagnostics.exitflag;
        
        if diagnostics.exitflag > 0
            nmpc_success_count = nmpc_success_count + 1;
        end
        
        if isfield(diagnostics, 'fallback_used') && diagnostics.fallback_used
            fallback_count = fallback_count + 1;
            fallback_used_history(k) = 1;
            if isfield(diagnostics, 'fallback_type')
                fallback_reason_history{k} = diagnostics.fallback_type;
            end
        end
        
        if ~isempty(diagnostics.predicted_oil)
            predicted_oil(k) = diagnostics.predicted_oil(1);
        end
        
    catch ME
        if config.verbose && k <= 5
            warning('Controller exception at k=%d: %s', k, ME.message);
        end
        u_opt = [actuators.BHP_actual; actuators.inj_actual];  % BHP in bar
        opt_time(k) = toc(t_opt_start);
        status_history(k) = -99;
        fallback_count = fallback_count + 1;
        fallback_used_history(k) = 1;
        fallback_reason_history{k} = 'EXCEPTION';
    end
    
    %% Apply DCS Veto
    for i = 1:4
        if ~dcs_policy.inj_enabled(i)
            u_opt(i+1) = dcs_policy.inj_min(i);
        end
    end
    
    %% Enforce Bounds (BHP in bar)
    u_opt(1) = max(base_constraints.BHP_min, min(u_opt(1), base_constraints.BHP_max));
    for i = 1:4
        u_opt(i+1) = max(dcs_policy.inj_min(i), min(u_opt(i+1), dcs_policy.inj_max(i)));
    end
    
    total_inj_commanded = sum(u_opt(2:5));
    if total_inj_commanded > dcs_policy.max_total_injection
        scale = dcs_policy.max_total_injection / total_inj_commanded;
        u_opt(2:5) = u_opt(2:5) * scale;
    end
    
    %% Apply Control
    actuators.BHP_command = u_opt(1);  
    actuators.inj_command = u_opt(2:5);
    BHP_command(k) = actuators.BHP_command;  
    inj_command(k, :) = actuators.inj_command';
    
    u_new_col = u_opt(:);
    upast_current = [upast_current(:, 2:end), u_new_col];
    uo_current = u_opt;
    
    %% Progress Reporting
    if config.verbose && (k <= 5 || mod(k, 20) == 0 || k == config.N_steps)
        fprintf('[Day %3d/%d] Oil: %5.0f | BHP: %5.1f bar | VRR: %.2f | Water: $%5.2f/BBL | NPV: $%6.2fM\n', ...
            k, config.N_steps, oil_noisy, actuators.BHP_actual, VRR_history(k), ...
            effective_water_cost_per_bbl(k), cumulative_npv(k)/1e6);
    end
end

sim_time = toc(sim_start_time);

%% ========== POST-SIMULATION ANALYSIS ==========
fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║          SIMULATION COMPLETE - FINAL RESULTS               ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

fprintf('Simulation time: %.1f minutes\n', sim_time/60);
fprintf('NMPC success rate: %.1f%%\n', 100*nmpc_success_count/nmpc_total_count);
fprintf('Fallback usage: %.1f%%\n\n', 100*fallback_count/nmpc_total_count);

fprintf('Production Results:\n');
fprintf('  Total oil: %.2f M STB\n', cumulative_oil/1e6);
fprintf('  Average rate: %.0f STB/day\n', mean(oil_measured));

fprintf('\nBHP Performance (bar):\n');
fprintf('  Mean:     %.1f bar\n', mean(BHP_actual));
fprintf('  Std Dev:  %.1f bar\n', std(BHP_actual));
fprintf('  Min:      %.1f bar\n', min(BHP_actual));
fprintf('  Max:      %.1f bar\n', max(BHP_actual));
fprintf('  Historical baseline: 100 bar (±5 bar)\n');
bhp_in_range = sum(BHP_actual >= 95 & BHP_actual <= 105) / config.N_steps * 100;
fprintf('  Time in historical range (95-105 bar): %.1f%%\n\n', bhp_in_range);

fprintf('Water Management:\n');
fprintf('  Total water: %.2f M BBL\n', cumulative_water/1e6);
fprintf('  Average VRR: %.2f\n', mean(VRR_history));
fprintf('  VRR violations (>4.0): %.1f%%\n\n', 100*sum(VRR_history > 4.0)/config.N_steps);

if water_cost_capped_count > 0
    fprintf('Water Cost Cap Impact:\n');
    fprintf('  Days capped: %d (%.1f%%)\n', water_cost_capped_count, 100*water_cost_capped_count/config.N_steps);
    total_water_volume = cumulative_water;
    avg_uncapped = water_cost_uncapped_total/water_cost_capped_count;
    water_volume_capped_days = total_water_volume * (water_cost_capped_count/config.N_steps);
    theoretical_uncapped_cost = water_volume_capped_days * avg_uncapped;
    actual_capped_cost = water_volume_capped_days * config.water_cost_max;
    true_savings = theoretical_uncapped_cost - actual_capped_cost;
    fprintf('  Total savings: $%.2fM\n\n', true_savings/1e6);
end

fprintf('Economic Analysis:\n');
fprintf('  Oil revenue:  $%.2f M\n', cumulative_revenue_discounted/1e6);
fprintf('  Water cost:   $%.2f M\n', cumulative_water_cost/1e6);
fprintf('  OPEX:         $%.2f M\n', cumulative_opex/1e6);
fprintf('  NPV:          $%.2f M\n\n', cumulative_npv(end)/1e6);

%% Store Results
results = struct();
results.time = time;
results.oil_measured = oil_measured;
results.oil_setpoint = oil_setpoint;
results.predicted_oil = predicted_oil;
results.BHP_actual = BHP_actual;  
results.BHP_command = BHP_command;  
results.inj_actual = inj_actual;
results.inj_command = inj_command;
results.VRR_history = VRR_history;
results.WOR_history = WOR_history;
results.water_cut_history = water_cut_history;
results.cumulative_npv = cumulative_npv;
results.daily_npv = daily_npv;
results.effective_water_cost_per_bbl = effective_water_cost_per_bbl;
results.opt_time = opt_time;
results.status_history = status_history;
results.marginal_cost_ratio_history = marginal_cost_ratio_history;
results.vrr_multiplier_history = vrr_multiplier_history;

% DCS data
results.dcs_active_injectors = dcs_active_injectors;
results.dcs_max_total_injection = dcs_max_total_injection;
results.dcs_water_multiplier = dcs_water_multiplier;
results.dcs_policy_history = dcs_policy_history;
results.dcs_vrr_target_history = dcs_vrr_target_history;
results.dcs_vrr_acceptable_history = dcs_vrr_acceptable_history;

% Fallback tracking
results.fallback_used_history = fallback_used_history;
results.fallback_reason_history = fallback_reason_history;

% Summary statistics
results.summary = struct();
results.summary.total_oil = cumulative_oil;
results.summary.total_water = cumulative_water;
results.summary.final_npv = cumulative_npv(end);
results.summary.avg_bhp_bar = mean(BHP_actual);
results.summary.std_bhp_bar = std(BHP_actual);
results.summary.bhp_in_historical_range_pct = bhp_in_range;
results.summary.avg_vrr = mean(VRR_history);
results.summary.nmpc_success_rate = nmpc_success_count / nmpc_total_count;
results.summary.fallback_rate = fallback_count / nmpc_total_count;

fprintf('📊 Results saved to ''results'' structure\n');
fprintf('   BHP values are in bar (native units)\n');
fprintf('   Historical baseline: 100 bar (±5 bar)\n\n');

%% Helper Function
function result = iif(cond, true_val, false_val)
    if cond, result = true_val; else, result = false_val; end
end

%% ============================================================================
%% plot_simulation_results.m - Comprehensive Plotting Script

if ~exist('results', 'var')
    error('Results structure not found! Run simulation first.');
end

fprintf('Creating comprehensive plots...\n');

%% Extract data from results
time = results.time;
oil_measured = results.oil_measured;
BHP_actual = results.BHP_actual;  
inj_actual = results.inj_actual;
VRR_history = results.VRR_history;
cumulative_npv = results.cumulative_npv;
effective_water_cost_per_bbl = results.effective_water_cost_per_bbl;

% Handle optional fields
if isfield(results, 'BHP_command')
    BHP_command = results.BHP_command;
else
    BHP_command = BHP_actual;
end

if isfield(results, 'water_cut_history')
    water_cut_history = results.water_cut_history;
else
    % Calculate from VRR if not available
    water_cut_history = VRR_history ./ (1 + VRR_history);
end

% Load additional variables if available
if isfield(results, 'oil_setpoint')
    has_setpoint = true;
    oil_setpoint = results.oil_setpoint;
else
    has_setpoint = false;
    oil_setpoint = oil_measured;
end

if isfield(results, 'inj_command')
    has_command = true;
    inj_command = results.inj_command;
else
    has_command = false;
    inj_command = inj_actual;
end

if isfield(results, 'opt_time')
    has_opt_time = true;
    opt_time = results.opt_time;
else
    has_opt_time = false;
end

if isfield(results, 'marginal_cost_ratio_history')
    has_mcr = true;
    marginal_cost_ratio_history = results.marginal_cost_ratio_history;
else
    has_mcr = false;
end

if isfield(results, 'dcs_max_total_injection')
    has_dcs = true;
    dcs_max_total_injection = results.dcs_max_total_injection;
else
    has_dcs = false;
end

if isfield(results, 'dcs_active_injectors')
    has_injector_status = true;
    dcs_active_injectors = results.dcs_active_injectors;
else
    has_injector_status = false;
end

if isfield(results, 'dcs_vrr_target_history')
    has_vrr_targets = true;
    dcs_vrr_target_history = results.dcs_vrr_target_history;
    dcs_vrr_acceptable_history = results.dcs_vrr_acceptable_history;
else
    has_vrr_targets = false;
end

if isfield(results, 'daily_npv')
    has_daily_npv = true;
    daily_npv = results.daily_npv;
else
    has_daily_npv = false;
end

%% Create main figure
fig = figure('Position', [50, 50, 1600, 900], 'Name', 'NMPC Results (BHP in bar)');

%% Subplot 1: Oil Production
subplot(4, 3, 1);
plot(time, oil_measured, 'b-', 'LineWidth', 2);
hold on;
if has_setpoint
    plot(time, oil_setpoint, 'r--', 'LineWidth', 1.5);
    legend('Measured', 'Setpoint', 'Location', 'best');
end
xlabel('Time (days)'); 
ylabel('Oil Rate (STB/d)');
title('Oil Production');
grid on;

%% Subplot 2: BHP 
subplot(4, 3, 2);
if has_command
    plot(time, BHP_command, 'r--', 'LineWidth', 1.5);
    hold on;
end
plot(time, BHP_actual, 'b-', 'LineWidth', 2);
xlabel('Time (days)'); 
ylabel('BHP (bar)');
title('Bottom Hole Pressure (bar)');
if has_command
    legend('Command', 'Actual', 'Location', 'best');
end
grid on;

%% Subplot 3: Injection Rates (Individual Lines)
subplot(4, 3, 3);
colors = {'b-', 'r-', 'g-', 'm-'};
hold on;
for i = 1:4
    plot(time, inj_actual(:, i), colors{i}, 'LineWidth', 2);
end
xlabel('Time (days)'); 
ylabel('Injection (BBL/d)');
title('Water Injection Rates');
legend('I1', 'I2', 'I3', 'I4', 'Location', 'best');
grid on;

%% Subplot 4: VRR
subplot(4, 3, 4);
plot(time, VRR_history, 'b-', 'LineWidth', 2);
hold on;
plot([time(1), time(end)], [4.0, 4.0], 'r--', 'LineWidth', 1.5);
xlabel('Time (days)'); 
ylabel('VRR (BBL/STB)');
title('Voidage Replacement Ratio');
legend('Actual', 'Target (4.0)', 'Location', 'best');
grid on;

%% Subplot 5: Water Cut
subplot(4, 3, 5);
plot(time, water_cut_history, 'b-', 'LineWidth', 2);
xlabel('Time (days)'); 
ylabel('Water Cut');
title('Water Cut');
grid on;

%% Subplot 6: Cumulative NPV
subplot(4, 3, 6);
plot(time, cumulative_npv/1e6, 'b-', 'LineWidth', 2);
xlabel('Time (days)'); 
ylabel('NPV ($M)');
title('Cumulative NPV (Discounted)');
grid on;

%% Subplot 7: Total Injection
subplot(4, 3, 7);
total_inj = sum(inj_actual, 2);
plot(time, total_inj, 'b-', 'LineWidth', 2);
hold on;
if has_dcs
    plot(time, dcs_max_total_injection, 'r--', 'LineWidth', 1.5);
    legend('Actual Total', 'DCS Limit', 'Location', 'best');
end
xlabel('Time (days)'); 
ylabel('Injection (BBL/d)');
title('Total Water Injection');
grid on;

%% Subplot 8: Injector Status
subplot(4, 3, 8);
if has_injector_status
    % Show injector status over time
    imagesc(time, 1:4, dcs_active_injectors');
    colormap(gca, [0.8 0.8 0.8; 0 0.5 0]);  % Gray for off, green for on
    colorbar('Ticks', [0, 1], 'TickLabels', {'Shut-in', 'Active'});
    xlabel('Time (days)');
    ylabel('Injector');
    set(gca, 'YTick', 1:4, 'YTickLabel', {'I1', 'I2', 'I3', 'I4'});
    title('Injector Status');
else
    % Show final injection distribution
    avg_inj = mean(inj_actual, 1);
    bar(avg_inj);
    set(gca, 'XTickLabel', {'I1', 'I2', 'I3', 'I4'});
    ylabel('Average Injection (BBL/d)');
    title('Injector Distribution');
    grid on;
end

%% Subplot 9: Water Cost
subplot(4, 3, 9);
plot(time, effective_water_cost_per_bbl, 'b-', 'LineWidth', 2);
hold on;
if exist('config', 'var')
    plot([time(1), time(end)], [config.water_cost_max, config.water_cost_max], ...
        'r--', 'LineWidth', 1.5);
    legend('Actual', sprintf('Cap ($%.0f/BBL)', config.water_cost_max), 'Location', 'best');
else
    plot([time(1), time(end)], [20, 20], 'r--', 'LineWidth', 1.5);
    legend('Actual', 'Cap ($20/BBL)', 'Location', 'best');
end
xlabel('Time (days)'); 
ylabel('Cost ($/BBL)');
title('Effective Water Cost (Capped)');
grid on;

%% Subplot 10: Optimization Time
subplot(4, 3, 10);
if has_opt_time
    plot(time, opt_time, 'b-', 'LineWidth', 2);
    xlabel('Time (days)'); 
    ylabel('Time (sec)');
    title('Optimization Time');
    grid on;
else
    text(0.5, 0.5, 'Optimization time data not available', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    axis off;
end

%% Subplot 11: Marginal Cost Ratio
subplot(4, 3, 11);
if has_mcr
    plot(time, marginal_cost_ratio_history, 'b-', 'LineWidth', 2);
    hold on;
    plot([time(1), time(end)], [0.7, 0.7], 'r--', 'LineWidth', 1);
    xlabel('Time (days)'); 
    ylabel('MCR');
    title('Marginal Cost Ratio (Econ Limit @0.7)');
    legend('Actual', 'Econ Limit', 'Location', 'best');
    grid on;
else
    text(0.5, 0.5, 'MCR data not available', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    axis off;
end

%% Subplot 12: VRR Management
subplot(4, 3, 12);
plot(time, VRR_history, 'b-', 'LineWidth', 2);
hold on;
if has_vrr_targets
    plot(time, dcs_vrr_target_history, 'g--', 'LineWidth', 1.5);
    plot(time, dcs_vrr_acceptable_history, 'r--', 'LineWidth', 1.5);
    legend('Actual', 'Target', 'Acceptable Max', 'Location', 'best');
else
    plot([time(1), time(end)], [2.5, 2.5], 'g--', 'LineWidth', 1.5);
    plot([time(1), time(end)], [4.0, 4.0], 'r--', 'LineWidth', 1.5);
    legend('Actual', 'Target (2.5)', 'Acceptable (4.0)', 'Location', 'best');
end
xlabel('Time (days)'); 
ylabel('VRR');
title('VRR Management');
grid on;

sgtitle('Hierarchical Economic NMPC Results (BHP in bar)', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Main plots created\n');

%% Create additional detailed figure
fig2 = figure('Position', [100, 100, 1400, 800], 'Name', 'Detailed Analysis (BHP in bar)');

%% Production Analysis
subplot(2, 3, 1);
plot(time, oil_measured, 'b-', 'LineWidth', 2);
hold on;
% Add 30-day moving average
if length(oil_measured) > 30
    ma30 = movmean(oil_measured, 30);
    plot(time, ma30, 'r--', 'LineWidth', 1.5);
    legend('Daily', '30-day MA', 'Location', 'best');
end
xlabel('Time (days)'); 
ylabel('Oil Rate (STB/d)');
title('Oil Production with Trend');
grid on;

%% BHP Analysis (in bar)
subplot(2, 3, 2);
histogram(BHP_actual, 30, 'FaceColor', 'b', 'EdgeColor', 'k');
xlabel('BHP (bar)'); 
ylabel('Frequency');
title('BHP Distribution (bar)');
grid on;

%% VRR Distribution
subplot(2, 3, 3);
histogram(VRR_history, 50, 'FaceColor', 'b', 'EdgeColor', 'k');
hold on;
xline(4.0, 'r--', 'LineWidth', 2, 'Label', 'Target');
xlabel('VRR'); 
ylabel('Frequency');
title('VRR Distribution');
grid on;

%% Injection Stacked Area
subplot(2, 3, 4);
area(time, inj_actual);
xlabel('Time (days)'); 
ylabel('Injection (BBL/d)');
title('Water Injection (Stacked)');
legend('I1', 'I2', 'I3', 'I4', 'Location', 'best');
grid on;

%% NPV Rate
subplot(2, 3, 5);
if has_daily_npv
    plot(time, daily_npv/1e3, 'b-', 'LineWidth', 2);
    hold on;
    plot([time(1), time(end)], [0, 0], 'r--', 'LineWidth', 1);
    xlabel('Time (days)'); 
    ylabel('Daily NPV ($K)');
    title('Daily NPV');
    grid on;
else
    % Calculate approximate daily NPV from cumulative
    daily_npv_approx = [0; diff(cumulative_npv)];
    plot(time, daily_npv_approx/1e3, 'b-', 'LineWidth', 2);
    hold on;
    plot([time(1), time(end)], [0, 0], 'r--', 'LineWidth', 1);
    xlabel('Time (days)'); 
    ylabel('Daily NPV ($K)');
    title('Daily NPV (Approximate)');
    grid on;
end

%% Water Cost vs VRR
subplot(2, 3, 6);
scatter(VRR_history, effective_water_cost_per_bbl, 20, time, 'filled');
colorbar;
xlabel('VRR'); 
ylabel('Water Cost ($/BBL)');
title('Water Cost vs VRR (colored by time)');
grid on;

sgtitle('Detailed Economic and Performance Analysis (BHP in bar)', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ Detailed analysis plots created\n');

%% Create BHP-specific analysis figure
fig3 = figure('Position', [150, 150, 1200, 600], 'Name', 'BHP Analysis (bar)');

%% BHP Time Series with Statistics
subplot(2, 3, 1);
plot(time, BHP_actual, 'b-', 'LineWidth', 2);
hold on;
if has_command
    plot(time, BHP_command, 'r--', 'LineWidth', 1.5);
end
mean_bhp = mean(BHP_actual);
std_bhp = std(BHP_actual);
plot([time(1), time(end)], [mean_bhp, mean_bhp], 'k--', 'LineWidth', 1);
xlabel('Time (days)'); 
ylabel('BHP (bar)');
title(sprintf('BHP: Mean=%.1f bar, Std=%.2f bar', mean_bhp, std_bhp));
if has_command
    legend('Actual', 'Command', 'Mean', 'Location', 'best');
else
    legend('Actual', 'Mean', 'Location', 'best');
end
grid on;

%% BHP vs Oil Rate
subplot(2, 3, 2);
scatter(BHP_actual, oil_measured, 20, time, 'filled');
colorbar;
xlabel('BHP (bar)'); 
ylabel('Oil Rate (STB/d)');
title('BHP vs Oil Production');
grid on;

%% BHP vs Total Injection
subplot(2, 3, 3);
total_inj = sum(inj_actual, 2);
scatter(BHP_actual, total_inj, 20, time, 'filled');
colorbar;
xlabel('BHP (bar)'); 
ylabel('Total Injection (BBL/d)');
title('BHP vs Total Injection');
grid on;

%% BHP Histogram with Statistics
subplot(2, 3, 4);
histogram(BHP_actual, 40, 'FaceColor', 'b', 'EdgeColor', 'k', 'Normalization', 'probability');
hold on;
xline(mean_bhp, 'r--', 'LineWidth', 2, 'Label', 'Mean');
xline(mean_bhp + std_bhp, 'g--', 'LineWidth', 1.5, 'Label', '+1σ');
xline(mean_bhp - std_bhp, 'g--', 'LineWidth', 1.5, 'Label', '-1σ');
xlabel('BHP (bar)'); 
ylabel('Probability');
title('BHP Distribution');
legend('Location', 'best');
grid on;

%% BHP Control Performance
subplot(2, 3, 5);
if has_command
    bhp_error = BHP_command - BHP_actual;
    plot(time, bhp_error, 'b-', 'LineWidth', 1.5);
    hold on;
    plot([time(1), time(end)], [0, 0], 'k--', 'LineWidth', 1);
    xlabel('Time (days)'); 
    ylabel('Error (bar)');
    title(sprintf('BHP Tracking Error (RMSE=%.2f bar)', rms(bhp_error)));
    grid on;
else
    text(0.5, 0.5, 'BHP command data not available', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    axis off;
end

%% BHP Rate of Change
subplot(2, 3, 6);
bhp_rate = [0; diff(BHP_actual)];
plot(time, bhp_rate, 'b-', 'LineWidth', 1.5);
hold on;
plot([time(1), time(end)], [0, 0], 'k--', 'LineWidth', 1);
xlabel('Time (days)'); 
ylabel('Rate of Change (bar/day)');
title('BHP Rate of Change');
grid on;

sgtitle('BHP Analysis (All Units in bar)', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('✓ BHP analysis plots created\n');

%% Print summary statistics
fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                  BHP STATISTICS (bar)                      ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
fprintf('  Mean:     %.2f bar\n', mean(BHP_actual));
fprintf('  Median:   %.2f bar\n', median(BHP_actual));
fprintf('  Std Dev:  %.2f bar\n', std(BHP_actual));
fprintf('  Min:      %.2f bar\n', min(BHP_actual));
fprintf('  Max:      %.2f bar\n', max(BHP_actual));
fprintf('  Range:    %.2f bar\n\n', range(BHP_actual));

if has_command
    fprintf('  Mean tracking error: %.2f bar\n', mean(BHP_command - BHP_actual));
    fprintf('  RMSE tracking error: %.2f bar\n\n', rms(BHP_command - BHP_actual));
end

fprintf('✓ All plots created successfully\n');
fprintf('  Figure 1: Main dashboard (12 subplots)\n');
fprintf('  Figure 2: Detailed analysis (6 subplots)\n');
fprintf('  Figure 3: BHP analysis (6 subplots)\n\n');