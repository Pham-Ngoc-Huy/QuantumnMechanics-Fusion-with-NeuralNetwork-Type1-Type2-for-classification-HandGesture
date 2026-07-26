function Y_defuzz = fuzzy_forward(X_b, C, Sigma, P, Q, K, num_features, num_class, num_rules, feat_of_rule)
    n_b = size(X_b, 2);

    % Layer 1: Fuzzification  →  MU (51 × 3 × batch)
    MU = zeros(num_features, K, n_b);
    for k = 1:K
        diff      = X_b - C(:,k);
        MU(:,k,:) = exp(-(diff.^2)./(2*Sigma(:,k).^2+1e-8));
    end

    % Layer 2: Rule firing  →  W_fire (153 × batch)
    W_fire = reshape(permute(MU,[2,1,3]), num_rules, n_b);

    % Layer 3: Normalization  →  W_norm (153 × batch)
    S      = sum(W_fire,1) + 1e-8;
    W_norm = W_fire ./ S;

    % Layer 4: Consequent  →  F_conseq (153 × 7 × batch), WF (153 × 7 × batch)
    X_feat   = X_b(feat_of_rule, :);   % (153 × batch)
    F_conseq = bsxfun(@times, reshape(X_feat, num_rules,1,n_b), ...
                      reshape(P, num_rules,num_class,1)) + ...
               reshape(Q, num_rules,num_class,1);
    WF = bsxfun(@times, reshape(W_norm, num_rules,1,n_b), F_conseq);

    % Layer 5: Defuzzification  →  Y_defuzz (num_class × batch)
    Y_defuzz = reshape(sum(WF,1), num_class, n_b);
end