function weights = getInjectorWeights(connMap, producer)
% GET_INJECTOR_WEIGHTS: Extract connectivity-based weights for injectors
%
% INPUTS:
%   connMap  : Connectivity map structure with fields:
%              .P1.inj{i}.name = 'I1', 'I2', etc.
%              .P1.inj{i}.strength = 0.0 to 1.0
%   producer : String or cell array, e.g., 'P1' or {'P1', 'P2'}
%
% OUTPUT:
%   weights  : [1 x 6] vector of normalized weights (0-1 scale)
%              Index i corresponds to injector Ii
%              Higher weight = stronger connectivity = more efficient
%
% EXAMPLE:
%   connMap = struct();
%   connMap.P1.inj{1}.name = 'I1'; connMap.P1.inj{1}.strength = 0.85;
%   connMap.P1.inj{2}.name = 'I2'; connMap.P1.inj{2}.strength = 0.45;
%   weights = getInjectorWeights(connMap, 'P1');
%   % weights = [1.0000, 0.5294, 0, 0, 0, 0] (normalized)

%% ========== INPUT VALIDATION ==========
if ischar(producer)
    producer = {producer};  % Convert to cell for uniform handling
end

%% ========== INITIALIZE WEIGHTS ==========
weights = zeros(1, 6);  % For I1-I6

%% ========== EXTRACT CONNECTIVITY FROM MAP ==========
for p = 1:length(producer)
    p_name = producer{p};
    
    % Check if producer exists in connMap
    if ~isfield(connMap, p_name)
        warning('ConnMap:ProducerNotFound', 'Producer %s not found in connectivity map.', p_name);
        continue;
    end
    
    % Check if injector list exists
    if ~isfield(connMap.(p_name), 'inj')
        warning('ConnMap:NoInjectors', 'No injectors found for producer %s.', p_name);
        continue;
    end
    
    % Iterate through all injectors for this producer
    n_injectors = length(connMap.(p_name).inj);
    
    for i = 1:n_injectors
        inj_info = connMap.(p_name).inj{i};
        
        % Extract injector name (e.g., 'I1', 'I2')
        if ~isfield(inj_info, 'name')
            warning('ConnMap:NoName', 'Injector %d for %s has no name field.', i, p_name);
            continue;
        end
        
        inj_name = inj_info.name;
        
        % Parse injector index from name
        inj_idx = parseInjectorIndex(inj_name);
        
        if isnan(inj_idx) || inj_idx < 1 || inj_idx > 6
            warning('ConnMap:InvalidIndex', 'Invalid injector name: %s', inj_name);
            continue;
        end
        
        % Extract connectivity strength
        if ~isfield(inj_info, 'strength')
            warning('ConnMap:NoStrength', 'Injector %s has no strength field.', inj_name);
            continue;
        end
        
        strength = inj_info.strength;
        
        % Validate strength range
        if strength < 0 || strength > 1
            warning('ConnMap:InvalidStrength', 'Strength %.2f for %s is out of range [0,1].', ...
                strength, inj_name);
            strength = max(0, min(1, strength));  % Clip to valid range
        end
        
        % Accumulate maximum strength across producers
        % (if injector supports multiple producers, use max connectivity)
        weights(inj_idx) = max(weights(inj_idx), strength);
    end
end

%% ========== NORMALIZE WEIGHTS ==========
% Normalize to [0, 1] range where 1 = strongest connection
max_weight = max(weights);

if max_weight > 0
    weights = weights / max_weight;
else
    % If no connectivity data, assume equal weights
    warning('ConnMap:NoData', 'No connectivity data found. Using equal weights.');
    weights = ones(1, 6);
end

%% ========== ENSURE MINIMUM WEIGHT ==========
% Avoid zero weights (causes issues in optimization)
% Set minimum to 0.2 (weakest injector still has 20% efficiency)
min_weight = 0.2;
weights = max(weights, min_weight);

% Re-normalize after applying floor
weights = weights / max(weights);

end

%% ========== HELPER: PARSE INJECTOR INDEX ==========
function idx = parseInjectorIndex(inj_name)
% Extract numeric index from injector name (e.g., 'I1' -> 1, 'INJECT-2' -> 2)

% Try standard format: 'I' followed by number
if startsWith(inj_name, 'I') || startsWith(inj_name, 'i')
    idx_str = extractAfter(inj_name, 1);  % Get substring after 'I'
    idx = str2double(idx_str);
    return;
end

% Try alternative formats: 'INJECT-1', 'INJ_2', etc.
digits = regexp(inj_name, '\d+', 'match');
if ~isempty(digits)
    idx = str2double(digits{1});
    return;
end

% If parsing fails, return NaN
idx = NaN;
end