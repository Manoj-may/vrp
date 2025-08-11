%% Complete Facility Location + Routing Optimization with MTZ integer `u` heuristic

clear; clc;

%% Parameters
N = 10; % Number of customers
K = 5;  % Number of facilities

%% Generate random data (fixed seed for reproducibility)
rng(1234);
x_customers = rand(N,1) * 10;
y_customers = rand(N,1) * 10;
x_init_facilities = rand(K,1) * 10;
y_init_facilities = rand(K,1) * 10;

%% Display input data
disp('Customer X coordinates:'); disp(x_customers');
disp('Customer Y coordinates:'); disp(y_customers');
disp('Initial facility X coords:'); disp(x_init_facilities');
disp('Initial facility Y coords:'); disp(y_init_facilities');

%% Define optimization variables indexing
nvar = N*K + K*K + K + 2*K; % total variables
idx_q = 1:N*K;
idx_r = N*K+1 : N*K+K*K;
idx_u = N*K+K*K+1 : N*K+K*K+K;
idx_yx = N*K+K*K+K+1 : N*K+K*K+K+K;
idx_yy = N*K+K*K+K+K+1 : nvar;

%% Bounds for variables
lb = [ zeros(N*K + K*K,1); ones(K,1); zeros(2*K,1) ];
ub = [ ones(N*K + K*K,1); K*ones(K,1); 10*ones(2*K,1) ];

%% Initial Guess
x0 = [
    repmat(1/K, N*K, 1);          % q: uniform initial assignment
    repmat(1/K, K*K, 1);          % r: uniform routing
    linspace(1,K,K)';             % u: initial MTZ orders 1..K
    x_init_facilities;            % yx: initial facility X coordinates
    y_init_facilities             % yy: initial facility Y coordinates
];

%% Constraint parameter
C0 = 5; % max total cost allowed

%% Objective function with penalty on `u` for integer and distinctness

function f = entropy_obj(x, N, K)
    q = reshape(x(1:N*K), N, K);
    r = reshape(x(N*K+1:N*K + K*K), K, K);
    u = x(N*K+K*K+1 : N*K+K*K+K);
    eps = 1e-6;
    
    q_clip = min(max(q, eps), 1 - eps);
    r_clip = min(max(r, eps), 1 - eps);
    
    % Entropy terms (encourage near-binary q, r)
    ent_term = sum(q_clip(:).*log(q_clip(:)) + (1 - q_clip(:)).*log(1 - q_clip(:))) + ...
               sum(r_clip(:).*log(r_clip(:)) + (1 - r_clip(:)).*log(1 - r_clip(:)));
           
    % Integer penalty: u close to integer
    integer_penalty = sum((u - round(u)).^2);
    
    % Distinctness penalty: penalize small differences between u elements
    dist_penalty = 0;
    for i = 1:K
        for j = i+1:K
            diff = u(i) - u(j);
            dist_penalty = dist_penalty + 1/(diff^2 + eps);
        end
    end
    
    % Weights for penalties
    w1 = 100;
    w2 = 100;
    
    f = ent_term + w1*integer_penalty + w2*dist_penalty;
end

%% Nonlinear constraints function

function [c, ceq] = nlcon(x, N, K, x_customers, y_customers, C0)
    q = reshape(x(1:N*K), N, K);
    r = reshape(x(N*K+1:N*K + K*K), K, K);
    u = x(N*K+K*K+1 : N*K+K*K+K);
    yx = x(N*K+K*K+K+1 : N*K+K*K+2*K);
    yy = x(N*K+K*K+2*K+1 : end);
    
    % Assignment cost (customer to facility distances)
    assign_cost = sum(sum(q .* ((x_customers - yx').^2 + (y_customers - yy').^2)));
    
    % Routing cost (facility to facility distances)
    route_cost = sum(sum(r .* ((yx - yx').^2 + (yy - yy').^2)));
    
    % Inequality constraint: total cost <= C0
    c1 = assign_cost + route_cost - C0;
    
    % Equality constraints:
    % 1) Each customer assigned exactly to one facility
    ceq1 = sum(q, 2) - 1;
    
    % 2) Routing constraints (no self loops)
    r_no_diag = r;
    r_no_diag(eye(K) == 1) = 0;
    
    % Outbound sum == 1 for each facility
    ceq2 = sum(r_no_diag, 1)' - 1;
    % Inbound sum == 1 for each facility
    ceq3 = sum(r_no_diag, 2) - 1;
    
    % No self-loop constraint (diagonal of r)
    ceq4 = diag(r);
    
    % MTZ sub-tour elimination constraints
    mtz = zeros(K*(K-1), 1);
    idx = 1;
    for l = 1:K
        for m_ = 1:K
            if l ~= m_
                mtz(idx) = u(l) - u(m_) + K * r(l,m_) - (K - 1);
                idx = idx + 1;
            end
        end
    end
    
    % Return inequalities and equalities
    c = [c1; mtz];
    ceq = [ceq1; ceq2; ceq3; ceq4];
end

%% Optimization options
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point', ...
                       'MaxFunctionEvaluations', 1e6);

%% Run fmincon optimization
[xsol, fval, exitflag, output] = fmincon(...
    @(x) entropy_obj(x, N, K), x0, [], [], [], [], lb, ub, ...
    @(x) nlcon(x, N, K, x_customers, y_customers, C0), options);

%% Extract solutions
q_sol = reshape(xsol(idx_q), [N, K]);
r_sol = reshape(xsol(idx_r), [K, K]);
u_sol = xsol(idx_u);
yx_sol = xsol(idx_yx);
yy_sol = xsol(idx_yy);

%% Round u to integers & fix duplicates if any
u_int = round(u_sol);

if numel(unique(u_int)) < K
    disp('Duplicates detected among rounded u. Fixing...');
    vals = 1:K;
    missing_vals = setdiff(vals, unique(u_int));
    counts = histcounts(u_int, 0.5:1:(K+0.5));
    duplicates = find(counts > 1);
    for d = duplicates
        idxs = find(u_int == d);
        for repl_idx = idxs(2:end)'
            if isempty(missing_vals)
                break;
            end
            u_int(repl_idx) = missing_vals(1);
            missing_vals(1) = [];
        end
    end
end

%% Display results
disp('Final assignment matrix (q):');
disp(q_sol);

disp('Final routing matrix (r):');
disp(r_sol);

disp('MTZ variables u (before rounding):');
disp(u_sol');

disp('MTZ variables u (after rounding and fixing duplicates):');
disp(u_int');

disp('Facility X locations (optimized):');
disp(yx_sol');

disp('Facility Y locations (optimized):');
disp(yy_sol');

disp('Initial facility X locations:');
disp(x_init_facilities');

disp('Initial facility Y locations:');
disp(y_init_facilities');
