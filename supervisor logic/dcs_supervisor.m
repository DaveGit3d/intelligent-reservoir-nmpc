function dcs_policy = dcs_supervisor(field_state, dcs_policy, base_constraints, verbose)
% DCS_SUPERVISOR - Distributed Control System for field-level constraint management
%
% FINAL VERSION with:
%   - Smooth VRR-responsive operational floor (prevents chattering)
%   - Relaxed BHP target (reduces infeasibility)
%   - Conservative initial VRR targets (prevents early over-injection)
%   - Connectivity-weighted injector allocation
%   - BHP UNITS: bar (native units, no psi conversion)
%   - Historical baseline: 100 bar (±5 bar)
%
% INPUTS:
%   field_state      - Current measurements (WOR, water_cut, avg_oil, trend, etc.)
%   dcs_policy       - Previous policy state (persistent)
%   base_constraints - Base operational limits (BHP in bar, injection bounds)
%   verbose          - Enable diagnostic output
%
% OUTPUTS:
%   dcs_policy - Updated policy (BHP_target in bar, inj_min, inj_max, VRR targets, etc.)

%% Initialize Policy
if isempty(dcs_policy)
    dcs_policy.BHP_target = 95;  % bar - RELAXED from 100 bar (historical baseline)
    dcs_policy.inj_enabled = ones(4, 1);
    dcs_policy.inj_min = base_constraints.inj_min;
    dcs_policy.inj_max = base_constraints.inj_max;
    dcs_policy.max_total_injection = base_constraints.max_total_injection;
    dcs_policy.water_cost_multiplier = 1.0;
    dcs_policy.policy_reason = 'INIT';
    dcs_policy.VRR_target = 3.0;          % Conservative initial target
    dcs_policy.VRR_acceptable_max = 5.5;  % Relaxed acceptable
    dcs_policy.current_injection_cap = base_constraints.max_total_injection;
    dcs_policy.clamp_counter = 0;
    dcs_policy.release_counter = 0;
end

%% Extract State
VRR = field_state.WOR;
wc = field_state.water_cut;
q_oil = field_state.avg_oil;
q_water = field_state.avg_water;
trend = field_state.oil_trend;
conn = field_state.injector_weights;

day = get_field(field_state, 'current_day', 0);
c_water = get_field(field_state, 'avg_water_cost', 7.5);

%% Phase-Adaptive VRR Targets (CONSERVATIVE EARLY, RELAXED LATER)
if q_oil > 8000
    if trend > -20
        phase = 'HIGH_PROD';
        dcs_policy.VRR_target = 2.8;      % Conservative (was 2.5)
        dcs_policy.VRR_acceptable_max = 4.5;
    else
        phase = 'DECLINE_HIGH';
        dcs_policy.VRR_target = 3.0;      % Conservative
        dcs_policy.VRR_acceptable_max = 5.0;
    end
elseif q_oil > 6000
    phase = 'MATURE';
    dcs_policy.VRR_target = 3.2;          % Slightly conservative (was 3.5)
    dcs_policy.VRR_acceptable_max = 5.5;
else
    phase = 'DECLINE_LATE';
    dcs_policy.VRR_target = 3.8;          % Conservative (was 4.0)
    dcs_policy.VRR_acceptable_max = 6.0;
end

if verbose
    fprintf('\n╔═══════════════════════════════════════════════════╗\n');
    fprintf('║ DCS UPDATE - %s (Day %d)\n', phase, day);
    fprintf('╠═══════════════════════════════════════════════════╣\n');
    fprintf('║ Oil: %.0f STB/d (trend: %+.0f) | Water: %.0f BBL/d\n', q_oil, trend, q_water);
    fprintf('║ VRR: %.2f (tgt: %.1f, max: %.1f) | WC: %.1f%%\n', ...
        VRR, dcs_policy.VRR_target, dcs_policy.VRR_acceptable_max, wc*100);
end

%% Production Priority Mode
production_priority = false;
if q_oil > 7000 && VRR < 3.2  % Tightened from 3.5
    production_priority = true;
    if verbose
        fprintf('║   📈 PRODUCTION PRIORITY MODE ACTIVE\n');
    end
end

%% Progressive Injection Control (RELAXED)
if production_priority
    VRR_upper = 6.5;  % Slightly reduced from 7.0
    VRR_lower = 2.2;  % Slightly increased from 2.0
else
    VRR_upper = 5.5;
    VRR_lower = 2.5;
end

if VRR > VRR_upper
    severity = (VRR - VRR_upper) / VRR_upper;
    
    if severity > 0.6
        factor = 0.95;
        label = 'SEVERE';
    elseif severity > 0.3
        factor = 0.97;
        label = 'MODERATE';
    else
        factor = 0.985;
        label = 'MILD';
    end
    
    dcs_policy.current_injection_cap = dcs_policy.current_injection_cap * factor;
    dcs_policy.clamp_counter = dcs_policy.clamp_counter + 1;
    dcs_policy.release_counter = 0;
    
    % Gentle BHP reduction (bar units)
    % Reduce by ~5 bar when VRR is high (relaxed from ~7 bar)
    dcs_policy.BHP_target = max(base_constraints.BHP_min + 2.0, ...
        dcs_policy.BHP_target - 5.0);  % bar reduction (was ~0.8 psi → ~5 bar)
    
    % Minimum BHP based on injection capacity (bar)
    min_bhp = base_constraints.BHP_min + 0.02 * (dcs_policy.current_injection_cap / 1000);
    if dcs_policy.BHP_target < min_bhp
        dcs_policy.BHP_target = min_bhp;
    end
    
    if VRR > 7.5
        [~, idx] = sort(conn, 'ascend');
        if conn(idx(1)) < 0.18 && VRR > 9.5
            dcs_policy.inj_enabled(idx(1)) = 0;
            dcs_policy.inj_max(idx(1)) = 0;
            msg = sprintf('%s CLAMP (%.0f%% cut) + SHUTIN I%d', label, (1-factor)*100, idx(1));
        else
            msg = sprintf('%s CLAMP (%.0f%% cut)', label, (1-factor)*100);
        end
    else
        msg = sprintf('%s CLAMP (%.0f%% cut)', label, (1-factor)*100);
    end
    
    if verbose
        fprintf('║   🚨 %s\n', msg);
        fprintf('║      Injection: %.0f BBL/d | BHP: %.1f bar\n', ...
            dcs_policy.current_injection_cap, dcs_policy.BHP_target);
    end
    
elseif VRR < VRR_lower
    factor = 1.06;
    dcs_policy.current_injection_cap = min(...
        dcs_policy.current_injection_cap * factor, ...
        base_constraints.max_total_injection);
    dcs_policy.release_counter = dcs_policy.release_counter + 1;
    dcs_policy.clamp_counter = 0;
    
    if verbose
        fprintf('║   🟢 RELEASE: VRR %.2f < %.1f (inject more)\n', VRR, VRR_lower);
        fprintf('║      Injection: %.0f BBL/d\n', dcs_policy.current_injection_cap);
    end
else
    % Stable - slow creep toward capacity
    if dcs_policy.current_injection_cap < 0.9 * base_constraints.max_total_injection
        factor = 1.01;
        dcs_policy.current_injection_cap = min(...
            dcs_policy.current_injection_cap * factor, ...
            base_constraints.max_total_injection);
    end
    
    if verbose && dcs_policy.clamp_counter > 0
        fprintf('║   🟡 STABLE: VRR in range (%.2f)\n', VRR);
    end
end

%% ========== SMOOTH VRR-RESPONSIVE OPERATIONAL FLOOR ==========
% Base floor from production rate
if production_priority
    base_floor = max(28000, 0.40 * base_constraints.max_total_injection);
elseif q_oil > 7000
    base_floor = max(22000, 0.35 * base_constraints.max_total_injection);
elseif q_oil > 5000
    base_floor = max(18000, 0.30 * base_constraints.max_total_injection);
else
    base_floor = max(15000, 0.25 * base_constraints.max_total_injection);
end

% SMOOTH VRR-based floor reduction (prevents chattering)
if VRR > 6.0
    % Critical VRR: emergency minimum
    operational_floor = max(8000, 0.15 * base_constraints.max_total_injection);
    if verbose
        fprintf('║   🚨 CRITICAL VRR (%.2f) - emergency floor: %.0f BBL/d\n', ...
            VRR, operational_floor);
    end
elseif VRR > 5.0
    % Smooth transition: VRR ∈ [5, 6] → floor ∈ [0.5*base, 8k]
    alpha = (VRR - 5.0) / (6.0 - 5.0);  % 0 → 1 as VRR: 5 → 6
    target_floor = (1 - 0.5*alpha) * base_floor;  % 1.0*base → 0.5*base
    operational_floor = max(8000, target_floor);
    
    if verbose && abs(VRR - 5.5) < 0.1  % Debug at midpoint
        fprintf('║   ⚠️  High VRR (%.2f) - smooth floor: %.0f BBL/d (alpha=%.2f)\n', ...
            VRR, operational_floor, alpha);
    end
elseif VRR > 4.0
    % Smooth transition: VRR ∈ [4, 5] → floor ∈ [0.7*base, 0.5*base]
    alpha = (VRR - 4.0) / (5.0 - 4.0);  % 0 → 1 as VRR: 4 → 5
    target_floor = (1 - 0.3*alpha) * base_floor;  % 1.0*base → 0.7*base
    operational_floor = max(12000, target_floor);
else
    % Normal VRR: use full base floor
    operational_floor = base_floor;
end

% Apply floor
if dcs_policy.current_injection_cap < operational_floor
    if verbose
        fprintf('║   ⚠️  FLOOR TRIGGERED: Cap %.0f → %.0f BBL/d (q_oil=%.0f, VRR=%.2f)\n', ...
            dcs_policy.current_injection_cap, operational_floor, q_oil, VRR);
    end
    dcs_policy.current_injection_cap = operational_floor;
end

dcs_policy.max_total_injection = max(dcs_policy.current_injection_cap, operational_floor);

%% Economic Limit Detection (LESS AGGRESSIVE)
mcr = get_field(field_state, 'marginal_cost_ratio', (VRR * c_water * 1.5) / 80);

if mcr > 0.85 && VRR > 7.0
    dcs_policy.water_cost_multiplier = min(4.0, dcs_policy.water_cost_multiplier * 1.15);
    
    if verbose
        fprintf('║   ⚠️  ECON WARNING: MCR=%.1f%%, VRR=%.2f\n', mcr*100, VRR);
    end
    
    if VRR > 9.0
        dcs_policy.max_total_injection = max(15000, operational_floor);
        dcs_policy.water_cost_multiplier = 4.0;
        if verbose
            fprintf('║   🆘 SURVIVAL MODE: Extreme VRR\n');
        end
    end
end

%% Water Cost Multiplier (LESS AGGRESSIVE)
policy = phase;

if VRR > 6.5
    dcs_policy.water_cost_multiplier = min(2.5, dcs_policy.water_cost_multiplier * 1.15);
    policy = [policy, '+VRR_HIGH'];
elseif VRR > 5.0
    dcs_policy.water_cost_multiplier = min(2.0, dcs_policy.water_cost_multiplier * 1.10);
    policy = [policy, '+VRR_MED'];
elseif VRR < 2.0
    dcs_policy.water_cost_multiplier = max(0.8, dcs_policy.water_cost_multiplier * 0.9);
    policy = [policy, '+VRR_LOW'];
end

if q_oil < 5000 && strcmp(phase, 'DECLINE_LATE')
    policy = [policy, '+LOW_RATE'];
    dcs_policy.water_cost_multiplier = min(3.0, dcs_policy.water_cost_multiplier * 1.08);
end

if wc > 0.92
    policy = [policy, '+HIGH_WC'];
    dcs_policy.water_cost_multiplier = min(3.5, dcs_policy.water_cost_multiplier * 1.15);
end

if contains(phase, 'DECLINE') && trend < -150
    policy = [policy, '+STEEP'];
    dcs_policy.water_cost_multiplier = min(2.5, dcs_policy.water_cost_multiplier * 1.08);
end

if VRR > dcs_policy.VRR_acceptable_max
    vrr_status = 'EXCESS';
elseif VRR > dcs_policy.VRR_target * 1.3
    vrr_status = 'MARGINAL';
elseif VRR < dcs_policy.VRR_target * 0.8
    vrr_status = 'EXCELLENT';
else
    vrr_status = 'NOMINAL';
end

dcs_policy.policy_reason = [phase, '_', vrr_status];

%% Connectivity-Based Allocation
n_active = sum(dcs_policy.inj_enabled);

if n_active > 0
    conn = conn(:);
    dcs_policy.inj_enabled = dcs_policy.inj_enabled(:);
    
    active_weights = conn .* dcs_policy.inj_enabled;
    wsum = sum(active_weights);
    
    if wsum > 1e-6
        min_total_feasible = 0.5 * dcs_policy.max_total_injection;
        
        for i = 1:4
            if dcs_policy.inj_enabled(i)
                frac = conn(i) / wsum;
                
                target_min = frac * min_total_feasible;
                dcs_policy.inj_min(i) = max(2000, min(target_min, base_constraints.inj_min(i)));
                
                target_max = frac * dcs_policy.max_total_injection;
                dcs_policy.inj_max(i) = min(target_max, base_constraints.inj_max(i));
                
                if dcs_policy.inj_max(i) < dcs_policy.inj_min(i) + 1000
                    dcs_policy.inj_max(i) = dcs_policy.inj_min(i) + 1000;
                end
            else
                dcs_policy.inj_min(i) = 0;
                dcs_policy.inj_max(i) = 0;
            end
        end
        
        total_min_required = sum(dcs_policy.inj_min);
        if total_min_required > dcs_policy.max_total_injection
            scale_factor = 0.95 * dcs_policy.max_total_injection / total_min_required;
            for i = 1:4
                if dcs_policy.inj_enabled(i)
                    dcs_policy.inj_min(i) = dcs_policy.inj_min(i) * scale_factor;
                    dcs_policy.inj_max(i) = max(dcs_policy.inj_max(i), dcs_policy.inj_min(i) + 500);
                end
            end
            
            if verbose
                fprintf('║   🔧 EMERGENCY SCALING: min sum %.0f → %.0f\n', ...
                    total_min_required, sum(dcs_policy.inj_min));
            end
        end
    else
        equal_share = dcs_policy.max_total_injection / n_active;
        for i = 1:4
            if dcs_policy.inj_enabled(i)
                dcs_policy.inj_min(i) = 0.35 * equal_share;
                dcs_policy.inj_max(i) = min(equal_share, base_constraints.inj_max(i));
            else
                dcs_policy.inj_min(i) = 0;
                dcs_policy.inj_max(i) = 0;
            end
        end
    end
end

%% Verbose Output (BHP in bar)
if verbose
    fprintf('║\n');
    fprintf('║ Policy: %s\n', dcs_policy.policy_reason);
    fprintf('║ Water mult: %.2fx | Envelope: %.0f BBL/d | Floor: %.0f BBL/d\n', ...
        dcs_policy.water_cost_multiplier, dcs_policy.max_total_injection, operational_floor);
    fprintf('║ BHP target: %.1f bar (historical baseline: 100 bar)\n', dcs_policy.BHP_target);
    fprintf('║\n');
    fprintf('║ Injectors (Connectivity-Weighted Bounds):\n');
    total_min = sum(dcs_policy.inj_min);
    total_max = sum(dcs_policy.inj_max);
    total_range = total_max - total_min;
    
    for i = 1:4
        icon = iif(dcs_policy.inj_enabled(i), '✅', '🔴');
        status = iif(dcs_policy.inj_enabled(i), 'ACTIVE ', 'SHUTIN ');
        if dcs_policy.inj_enabled(i)
            pct_min = 100 * dcs_policy.inj_min(i) / max(total_min, 1);
            pct_max = 100 * dcs_policy.inj_max(i) / max(total_max, 1);
            range_i = dcs_policy.inj_max(i) - dcs_policy.inj_min(i);
            fprintf('║   %s I%d: %s | Min: %5.0f (%.1f%%) | Max: %5.0f (%.1f%%) | Range: %5.0f | W: %.2f\n', ...
                icon, i, status, dcs_policy.inj_min(i), pct_min, ...
                dcs_policy.inj_max(i), pct_max, range_i, conn(i));
        else
            fprintf('║   %s I%d: %s | W: %.2f\n', icon, i, status, conn(i));
        end
    end
    
    fprintf('║   Total: Min=%.0f | Max=%.0f | Range=%.0f BBL/d\n', ...
        total_min, total_max, total_range);
    
    if total_range < 10000
        fprintf('║   ⚠️  WARNING: Narrow feasible range (%.0f BBL/d)\n', total_range);
    end
    
    fprintf('╚═══════════════════════════════════════════════════╝\n\n');
end

end

%% Helper Functions
function val = get_field(s, f, def)
if isfield(s, f), val = s.(f); else, val = def; end
end

function r = iif(c, t, f)
if c, r = t; else, r = f; end
end