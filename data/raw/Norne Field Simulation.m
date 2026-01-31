%% Norne Field Simulation 
% This script loads the Norne field model from local .DATA/.GRDECL files,
% attempts to handle multi-component grid issues based on detected behavior,
% runs a black-oil simulation, and extracts key well outputs.
clear all; clc;

%% Initialize Environment
mrstModule add ad-core ad-blackoil ad-props mrst-gui deckformat linearsolvers
gravity reset on;    % Enable gravity
mrstVerbose on;      % Turn on verbose output
useMex = true;       % Use compiled C++ accelerators when available

%% Set Path to Local Norne Data
data = 'opm-data-master/norne';
fprintf('Attempting to load Norne data from: %s\n', fullfile(pwd, data));

%% Load Norne Dataset from Local Directory
try
    deck = readEclipseDeck(fullfile(data, 'NORNE_ATW2013.DATA'));
    deck = convertDeckUnits(deck); % Convert to MRST units
    grdecl = readGRDECL(fullfile(data, 'NORNE_ATW2013.DATA'));
    % Use the actual ACTNUM from the deck data if available
    if isfield(deck.GRID, 'ACTNUM')
        fprintf('FOUND ACTNUM\n');
        grdecl.ACTNUM = deck.GRID.ACTNUM;
    else
        fprintf('Warning: ACTNUM not found in deck. Proceeding with GRDECL ACTNUM or default.\n');
    end
    G_processed = processGRDECL(grdecl, 'checkgrid', false);
catch ME
    fprintf(2, 'Error loading deck or grdecl files:\n%s\n', ME.message);
    return; % Stop execution if loading fails
end

%% Handle Multiple Connected Components
% ... [Existing code for grid processing] ...
% The Norne grid data contains multiple disconnected components. Your MRST version's
% computeGeometry might not handle a single grid struct containing multiple components.
% We check the output of processGRDECL.

% %% --- Handle Multiple Connected Components ---
%
G = [];  % Initialize G
disp(G_processed)
if isstruct(G_processed)
    fprintf('processGRDECL returned a cell array of %d grid components.\n', numel(G_processed));

    if ~isempty(G_processed)
        % Ensure each cell has a valid structure with 'cells.num'
        num_cells_in_comps = zeros(1, numel(G_processed));
        for i = 1:numel(G_processed)
            gi = G_processed(i);
            
            if  isfield(gi, 'cells') && isstruct(gi.cells) && isfield(gi.cells, 'num')
                num_cells_in_comps(i) = gi.cells.num;
            else
                fprintf(2, 'Warning: Grid component %d is invalid or missing cells.num\n', i);
            end
        end

        [~, largest_comp_idx] = max(num_cells_in_comps);
        if num_cells_in_comps(largest_comp_idx) > 0
            G = G_processed(largest_comp_idx);
            fprintf('Selected component %d with %d cells for simulation.\n', ...
                largest_comp_idx, G.cells.num);
        else
            error('No valid grid components with cells.num found.');
        end
    else
        error('processGRDECL returned an empty cell array.');
    end

    else
        error('processGRDECL returned an unsupported type (neither cell nor struct).');
    end
    
    % Final check on grid G before proceeding
    if isempty(G)
        error('Processed grid G is empty.');
    end
    
    if ~isfield(G, 'cells')
        error('Processed grid G does not contain ''cells'' field.');
    end
    if ~isstruct(G.cells)
        error('Processed grid G.cells is not a struct.');
    end
    
    if ~isfield(G.cells, 'num')
        error('Processed grid G.cells is missing ''num'' field.');
    end
    
    if G.cells.num == 0
        error('Processed grid G has zero cells.');
    end

%% 
% %% --- Prepare grid G for computeGeometry ---
% % Ensure G.cells.centroids field exists before computeGeometry,
% % as it seemed to cause an error previously when using 'subgrid'.
% % This might still be necessary even if G came directly from processGRDECL.
% % Only attempt to initialize if G.cells is a valid struct with cells
if isfield(G, 'cells') && isstruct(G.cells) && isfield(G.cells, 'num') && G.cells.num > 0
    if ~isfield(G.cells, 'centroids')
         G.cells.centroids = zeros(G.cells.num, 3); % Initialize with zeros
         fprintf('Initialized G.cells.centroids field before computeGeometry.\n');
    end
else
     fprintf(2, 'Warning: G.cells is not a valid struct for centroids initialization.\n');
     % computeGeometry will likely fail here if G is invalid or lacks cells
end
%
%
% %% --- Now compute geometry ---
% % This line caused the initial error about multiple components when G was
% % a single struct with multiple internal components.
% % If G was selected from a cell array, this should work.
% % If G is a single struct (because processGRDECL didn't split), this might
% % fail again due to the internal components, as we couldn't split it.
fprintf('Attempting to compute geometry...\n');
try
        computed_G = computeGeometry(G);
        G = computed_G;
        fprintf('Computed geometry for the grid.\n');

catch ME
    fprintf(2, 'Error computing geometry:\n%s\n', ME.message);
    fprintf(2, 'This error likely means the grid contains multiple components that computeGeometry cannot handle.\n');
    fprintf(2, 'Your MRST version does not have the ''partition_components'' function needed to split the grid.\n');
    fprintf(2, 'Consider updating your MRST installation or look for alternative grid processing functions in your specific version.\n');
    return; % Stop execution if geometry computation fails
end

%% Rock and Fluid Properties
try
    % Initialize rock properties from the deck. MRST should map these to G.
    rock = initEclipseRock(deck);
    if isfield(G.cells, 'indexMap') % Use indexMap if available
       rock = compressRock(rock, G.cells.indexMap); % Compress rock properties to match G
    else
       % Fallback if indexMap is missing (less robust)
       fprintf(2, 'Warning: G.cells.indexMap missing. Skipping rock compression.\n');
    end


    % Handle zero/near-zero porosity cells - Apply this to the grid rock
    min_poro = 1e-6;
    if isfield(rock, 'poro') && ~isempty(rock.poro)
        rock.poro(rock.poro < min_poro) = min_poro;
    else
        fprintf(2, 'Warning: Rock structure is missing or has empty ''poro'' field.\n');
    end
    fprintf('Initialized rock properties.\n');


    % Initialize blackoil fluid from the deck with optional MEX acceleration
    % Suppress warnings about unsupported VFP or Hysteresis which are common for Norne
    warning('off', 'mrst:unsupportedVFP');
    warning('off', 'mrst:unsupportedHysteresis');
    fluid = initDeckADIFluid(deck, 'useMex', useMex);
    warning('on', 'mrst:unsupportedVFP');
    warning('on', 'mrst:unsupportedHysteresis');

    fprintf('Initialized fluid properties.\n');

catch ME
     fprintf(2, 'Error initializing rock or fluid properties:\n%s\n', ME.message);
     return;
end

    disp('Grid G:');
disp(G);

disp('Rock properties:');
disp(rock);
% Check fluid properties
disp('Fluid PVT properties:');
disp(structfun(@(x) class(x), fluid, 'UniformOutput', false));


%% Blackoil Model Configuration (Updated Validation)
try
    % Create base model
    model = ThreePhaseBlackOilModel(G, rock, fluid);
    
    % Set phases directly (more compatible approach)
    model.oil = isfield(deck.RUNSPEC, 'OIL') && deck.RUNSPEC.OIL;
    model.water = isfield(deck.RUNSPEC, 'WATER') && deck.RUNSPEC.WATER;
    model.gas = isfield(deck.RUNSPEC, 'GAS') && deck.RUNSPEC.GAS;
    
    % Set special features
    model.disgas = isfield(deck.RUNSPEC, 'DISGAS') && deck.RUNSPEC.DISGAS;
    model.vapoil = isfield(deck.RUNSPEC, 'VAPOIL') && deck.RUNSPEC.VAPOIL;
    
    % Initialize operators
    model.operators = setupOperatorsTPFA(G, rock);
    
    % ===== COMPATIBLE VALIDATION =====
    % Skip formal validation if method doesn't exist
    if ~isempty(which('validateModel'))
        try
            model = model.validateModel();
        catch
            fprintf('Using simplified validation (older MRST version)\n');
            % Manually set required properties
            model.FacilityModel = FacilityModel(model);
            model.FlowPropertyFunctions = FlowPropertyFunctions(model);
        end
    else
        fprintf('Using minimal initialization (pre-validation MRST)\n');
        model.FacilityModel = FacilityModel(model);
    end
    
    % ===== COMPONENTS WORKAROUND =====
      % Check if model.Components is empty and fill if needed
if isempty(model.Components) || (iscell(model.Components) && all(cellfun(@isempty, model.Components)))
    fprintf('Warning: model.Components is empty. Initializing components...\n');
    
    % Initialize components structure
    phases = {};
    
    % Check for active phases in RUNSPEC
    if isfield(deck.RUNSPEC, 'OIL') && deck.RUNSPEC.OIL
        phases{end+1} = struct('name', 'Oil', 'phase', 'O', 'id', numel(phases)+1, ...
                              'density', [], 'viscosity', [], 'pvt', []);
    end
    if isfield(deck.RUNSPEC, 'WATER') && deck.RUNSPEC.WATER
        phases{end+1} = struct('name', 'Water', 'phase', 'W', 'id', numel(phases)+1, ...
                              'density', [], 'viscosity', [], 'pvt', []);
    end
    if isfield(deck.RUNSPEC, 'GAS') && deck.RUNSPEC.GAS
        phases{end+1} = struct('name', 'Gas', 'phase', 'G', 'id', numel(phases)+1, ...
                              'density', [], 'viscosity', [], 'pvt', []);
    end
    
    % Validate we have at least one phase
    if isempty(phases)
        error('No active phases found in RUNSPEC (OIL, WATER, GAS all missing or zero).');
    end
    
    % Assign to model
    model.Components = phases;
    
    % Log the initialization
    phaseNames = cellfun(@(x) x.name, phases, 'UniformOutput', false);
    fprintf('Initialized Components with phases: %s\n', strjoin(phaseNames, ', '));
    
 
end
    
    % Display configuration
    fprintf('\n=== Model Configuration ===\n');
    fprintf('Phases: Oil(%d), Water(%d), Gas(%d)\n', model.oil, model.water, model.gas);
    fprintf('Features: DISGAS(%d), VAPOIL(%d)\n', model.disgas, model.vapoil);
    fprintf('- Active components: %d\n', model.getNumberOfComponents());
    fprintf('Operators: %d transmissibilities\n', numel(model.operators.T));
    

catch ME
    fprintf(2, '\nCONFIGURATION ERROR: %s\n', ME.message);
    fprintf(2, 'Recommended actions:\n');
    fprintf(2, '1. Verify MRST version with: mrstVersion()\n');
    fprintf(2, '2. Check model class with: class(model)\n');
    fprintf(2, '3. Try GenericBlackOilModel instead\n');
    
    % Fallback to simpler model if available
    try
        fprintf('\nAttempting GenericBlackOilModel...\n');
        model = GenericBlackOilModel(G, rock, fluid);
        model = model.setupFromDeck(deck);
        fprintf('Successfully created GenericBlackOilModel\n');
    catch
        fprintf(2, 'Fallback model also failed\n');
    end
    
    return;
end

%% Simulation Initialization with Advanced Well Classification
% ... [Existing code for well classification] ...
try
    fprintf('Initializing simulation state and schedule with enhanced well classification...\n');
    
    % 1. Model configuration check
    if ~isfield(model, 'FacilityModel')
        model.FacilityModel = FacilityModel(model);
        fprintf('Added FacilityModel to handle wells\n');
    end
    
    % 2. Initialize state with relperm warning handling
    warning('off', 'mrst:initState:relperm');
    state0 = initStateDeck(model, deck);
    warning('on', 'mrst:initState:relperm');
    fprintf('Initial state created with %d cells\n', G.cells.num);
    
    % 3. Initialize and validate schedule
    if ~isfield(deck, 'SCHEDULE') || isempty(deck.SCHEDULE)
        error('Deck missing required SCHEDULE section');
    end
    
    schedule = convertDeckScheduleToMRST(model, deck);
    
    % 4. Create enhanced well data logging structure
    simData = struct();
    simData.wellLog = struct(...
        'timestamp', datetime('now'), ...
        'modelType', class(model), ...
        'wellClassificationMethod', 'controlTargets+compi');
    simData.wellLog.wells = struct();
    
    % 5. Process and classify each well
    if isempty(schedule.control(1).W)
        error('No wells initialized in schedule');
    end
    
    fprintf('\nProcessing %d wells...\n', numel(schedule.control(1).W));
    validWells = true(numel(schedule.control(1).W), 1);
    
    for i = 1:numel(schedule.control(1).W)
        well = schedule.control(1).W(i);
        wellName = well.name;
        simData.wellLog.wells(i).name = wellName;
        simData.wellLog.wells(i).originalType = well.type;
        
        try
            % Standardize cells data
            if ~isfield(well, 'cells') || isempty(well.cells)
                well.cells = zeros(0, 1);
            elseif ischar(well.cells) || isstring(well.cells)
                well.cells = str2num(well.cells);
            elseif iscell(well.cells)
                well.cells = cell2mat(well.cells);
            end
            well.cells = double(well.cells(:)); % Force column vector
            
            % Enhanced well classification
            [wellType, classificationInfo] = classifyWell(well, model);
            well.type = wellType;
            
            % Log classification results
            simData.wellLog.wells(i).classifiedType = wellType;
            simData.wellLog.wells(i).classificationInfo = classificationInfo;
            simData.wellLog.wells(i).cellsSize = size(well.cells);
            simData.wellLog.wells(i).cellsSample = well.cells(1:min(3,end));
            
            % Validate based on classified type
            if strcmpi(wellType, 'prod') || strcmpi(wellType, 'inj')
                simData.wellLog.wells(i).status = 'valid';
                if isempty(well.cells)
                    warning('Well %s has empty cells data', wellName);
                    simData.wellLog.wells(i).status = 'empty_data';
                end
            else
                validWells(i) = false;
                simData.wellLog.wells(i).status = 'unclassified';
                warning('Well %s could not be classified', wellName);
            end
            
            % Store cleaned well data back
            schedule.control(1).W(i) = well;
            
        catch ME
            validWells(i) = false;
            simData.wellLog.wells(i).status = 'error';
            simData.wellLog.wells(i).errorMsg = ME.message;
            warning('Error processing well %s: %s', wellName, ME.message);
        end
    end
    
    % Remove invalid/unclassified wells
    if any(~validWells)
        removedNames = {schedule.control(1).W(~validWells).name};
        fprintf('Removing %d invalid/unclassified wells:\n', sum(~validWells));
        fprintf('  %s\n', removedNames{:});
        schedule.control(1).W = schedule.control(1).W(validWells);
        simData.wellLog.removedWells = removedNames;
    end
    
    % 6. Final reporting
    prodCount = sum(strcmpi({schedule.control(1).W.type}, 'prod'));
    injCount = sum(strcmpi({schedule.control(1).W.type}, 'inj'));
    
    fprintf('\nInitialization completed successfully:\n');
    fprintf('- Grid: %d cells, avg pressure %.2f bar\n', ...
            G.cells.num, mean(state0.pressure)/barsa);
    fprintf('- Wells: %d valid (%d prod, %d inj)\n', ...
            numel(schedule.control(1).W), prodCount, injCount);
    
    % Store the enhanced logging structure
    schedule.simData = simData;
    
catch ME
    fprintf(2, '\nINITIALIZATION FAILED:\n%s\n', ME.message);
    if exist('simData', 'var')
        simData.error = ME.message;
        schedule.simData = simData;
    end
    rethrow(ME);
end

%% Helper function for well classification
function [wellType, info] = classifyWell(well, model)
    info = struct();
    originalType = well.type;
    
    % Check for standard types first
    if any(strcmpi(originalType, {'prod', 'injector', 'inj', 'producer'}))
        wellType = lower(originalType);
        if startsWith(wellType, 'prod')
            wellType = 'prod';
        else
            wellType = 'inj';
        end
        info.method = 'standard_type';
        return;
    end
    
    % Classification based on control targets
    if isfield(well, 'val') && ~isempty(well.val)
        target = well.val(1); % First control target
        info.targetValue = target;
        
        if target < 0
            wellType = 'prod';
            info.method = 'negative_target';
            return;
        elseif target > 0
            wellType = 'inj';
            info.method = 'positive_target';
            return;
        end
    end
    
    % Classification based on composition (compi)
    if isfield(well, 'compi') && ~isempty(well.compi)
        compi = well.compi;
        info.composition = compi;
        
        % Check for injection signature (high water/gas fraction)
        if isnumeric(compi) && length(compi) >= 2
            if compi(2) > 0.8 % Water injection
                wellType = 'inj';
                info.method = 'water_composition';
                return;
            elseif isnumeric(compi) && length(compi) >= 3 && compi(3) > 0.8 % Gas injection
                wellType = 'inj';
                info.method = 'gas_composition';
                return;
            end
        end
        
        % Check for production signature (high oil fraction)
        if length(compi) >= 1 && compi(1) > 0.5
            wellType = 'prod';
            info.method = 'oil_composition';
            return;
        end
    end
    
    % Classification based on well sign if present
    if isfield(well, 'sign') && ~isempty(well.sign)
        info.wellSign = well.sign;
        if well.sign > 0
            wellType = 'inj';
            info.method = 'positive_sign';
            return;
        else
            wellType = 'prod';
            info.method = 'negative_sign';
            return;
        end
    end
    
    % Final fallback for RESV_HISTORY, rate, orat types
    switch lower(originalType)
        case {'orat', 'rate'}
            % Typically production metrics
            wellType = 'prod';
            info.method = 'type_name_fallback';
        case 'resv_history'
            % Typically monitoring wells - exclude from simulation
            wellType = 'monitor';
            info.method = 'reservoir_history';
        otherwise
            wellType = originalType;
            info.method = 'unclassified';
    end
end
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

%% Nonlinear Solver Configuration
nls = NonLinearSolver();
nls.maxIterations = 18;
nls.maxTimestepCuts = 10;
nls.useRelaxation = true;
nls.minRelaxation = 0.5;

model.toleranceCNV = 1e-2;
model.toleranceMB = 1e-7;

%% Run Simulation
try
    [wellSols, states, report] = simulateScheduleAD(state0, model, schedule, 'NonLinearSolver', nls);
catch ME
    fprintf(2, 'Error during simulation:\n%s\n', ME.message);
end

%% Post-Processing: Calculate Time Vector
T = [];
if ~isempty(wellSols) && isfield(schedule, 'step') && ~isempty(schedule.step)
    last_sol_idx = numel(wellSols);
    if last_sol_idx > 0 && last_sol_idx <= numel(schedule.step.val)
        T = cumsum(schedule.step.val(1:last_sol_idx));
    end
end

%% --- Plotting Field-Level Results ---
% ... [Existing code for plotting] ...

%% Add Incompressible Flow Simulation Setup
% Structure to hold the reservoir state
sol = initResSol(G, 0.0);

% Pressure 100 bar at top of the column, no flow (v=0)
bc = pside([], G, 'TOP', 100.*barsa());

% Transmissibility coefficient of proportionality Tij
hT = computeTrans(G, rock);

% Fluid properties: incompressible flow
mrstModule add incomp
gravity reset on
fluid_incomp = initSingleFluid('mu', 1*centi*poise, 'rho', 1014*kilogram/meter^3);
[mu, rho] = fluid_incomp.properties();

% Incompressible solver
sol_incomp = incompTPFA(sol, G, hT, fluid_incomp, 'bc', bc);

% Extract results and plot
% ... [Code to visualize incompressible results] ...

fprintf('Incompressible flow simulation completed.\n');