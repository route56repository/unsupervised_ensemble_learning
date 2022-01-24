function [min_k,cfun] = algorithm1(Covmat)
%function [min_k,res,cfun,v_on,v_off,X_on,X_off] = algorithm1(Covmat)

R = size(Covmat,1); % No of. classifiers

%% Score matrix generation (eq. 15, Jaffe 2016):
S = scorematrix(Covmat);

%% c function estimation & matrix completion of R_on / R_off:
% Empty cell arrays and matrices are created for subsequent result history record after for-loop.
resvec = 1e2.*ones(R,1); cfun_cell = cell(1,R); idS_mat = zeros(R); 
M_on_cell = cell(1,R); M_off_cell = cell(1,R);
v_on_mat = zeros(R); v_off_mat = zeros(R);
X_on_cell = cell(1,R); y_on_cell = cell(1,R); Z_on_cell = cell(1,R);
X_off_cell = cell(1,R); y_off_cell = cell(1,R); Z_off_cell = cell(1,R);

%{
resvec = ones(R-1,1); cfun_cell = cell(1,R-1); idS_mat = zeros(R,R-1); 
M_on_cell = cell(1,R-1); M_off_cell = cell(1,R-1);
v_on_mat = zeros(R,R-1); v_off_mat = zeros(R,R-1);
X_on_cell = cell(1,R-1); y_on_cell = cell(1,R-1); Z_on_cell = cell(1,R-1);
X_off_cell = cell(1,R-1); y_off_cell = cell(1,R-1); Z_off_cell = cell(1,R-1);
%}

% The following for-loop iterates for all possible no. of correlated sub-groups of classifiers (k):
%for k = 1:R 
for k = 2:R
    % Spectral clustering is performed on score matrix S:
    idS = spectralcluster(S,k,'Distance','precomputed'); % Line 4 of Alg. 1, Jaffe 2016
    % NOTE: It is assumed that S is a similarity matrix 
    % Output Rx1 vector idS assigns a correlation sub-group to each classifier
    
    %idS = spectral_cluster(S,k,1,0);

    % M_on / M_off are matrices recording indices and values of known
    % entries of R_on / R_off rank-one matrices (see eqs. 10, 11 & 12, Jaffe 2016)
    M_on = []; M_off = []; % Dimension: m_on x 3 / m_off x 3. Structure: (i index, j index, entry value).
    cfun = eye(R); % c indication function (eq. 11, Jaffe 2016)
    for ii = 1:R
        for jj = ii+1:R % symmetric matrix
            if idS(ii) == idS(jj) % if classifier ii belongs to the same group as jj's...
                cfun(ii,jj) = 1; cfun(jj,ii) = 1; % ...assign 1 to cfun's (ii,jj) entry
                M_on = [M_on; ii jj Covmat(ii,jj)]; % record non-diagonal values only in M_on
            elseif idS(ii)~= idS(jj) % for the opposed case (and ensuring non-diagonal values again)...
                M_off = [M_off; ii jj Covmat(ii,jj)]; %... record values in M_off
            end
        end
    end
    clear ii jj
    
    if size(M_on,1) > 0 && size(M_off,1) > 0
        % M_on / M_off: matrix completion via trace minimization (see p. 49, Candes 2009)
        % NOTE: see 'tracemin' function script for further info
       [v_on,X_on,y_on,Z_on] = tracemin(R,M_on);
       [v_off,X_off,y_off,Z_off] = tracemin(R,M_off);
    elseif size(M_off,1) == 0
        X_on = estimate_rank_1_matrix(Covmat); X_off = [];
        v_on = diag(X_on); y_on = []; Z_on = []; 
        v_off = zeros(R,1); y_off = []; Z_off = [];
    elseif size(M_on,1) == 0
        X_off = estimate_rank_1_matrix(Covmat); X_on = [];
        v_off = diag(X_off); y_off = []; Z_off = [];
        v_on = zeros(R,1); y_on = []; Z_on = [];
    end
  
    % Residual computation (eq. 14, Jaffe 2016):
    res_ij = 0;
    for ii = 1:R
        for jj = [(1:1:ii-1) (ii+1:1:R)] % i index skipped
            op_ij = (cfun(ii,jj)*(v_on(ii)*v_on(jj)-Covmat(ii,jj))^2)+((1-cfun(ii,jj))*(v_off(ii)*v_off(jj)-Covmat(ii,jj))^2);
            res_ij = res_ij + op_ij;
        end
    end
    resvec(k) = res_ij;
    cfun_cell{k} = cfun; idS_mat(:,k) = idS;
    M_on_cell{k} = M_on; M_off_cell{k} = M_off;
    v_on_mat(:,k) = v_on; v_off_mat(:,k) = v_off;
    X_on_cell{k} = X_on; y_on_cell{k} = y_on; Z_on_cell{k} = Z_on;
    X_off_cell{k} = X_off; y_off_cell{k} = y_off; Z_off_cell{k} = Z_off;
end
clear k ii jj op_ij res_ij
[res, min_k] = min(resvec); % minimum residual (res) and respective index location (min_k)

%% Final results:
cfun = cfun_cell{min_k}; 

idS = idS_mat(:,min_k);
v_on = v_on_mat(:,min_k); v_off = v_off_mat(:,min_k);
X_on = X_on_cell{min_k}; y_on = y_on_cell{min_k}; Z_on = Z_on_cell{min_k};
X_off = X_off_cell{min_k}; y_off = y_off_cell{min_k}; Z_off = Z_off_cell{min_k};
end