function [c, ceq] = production_constraint(u_vec, Np, Nu, ypast, upast, ...
    nlarx_model, scaling_params, min_prod, bias_correction)
% Nonlinear constraint: average production >= min_prod

n_inputs = 5;
u_seq = reshape(u_vec, n_inputs, Nu);
u_full = [u_seq, repmat(u_seq(:,end), 1, Np-Nu)];

[y_pred, ~] = predict_surrogate_model(nlarx_model, u_full, ypast, ...
    upast, scaling_params, 0);

y_pred = y_pred + bias_correction;

avg_production = mean(y_pred);

% Inequality constraint: c <= 0
c = min_prod - avg_production;  % If avg < min, constraint violated

% No equality constraints
ceq = [];
end