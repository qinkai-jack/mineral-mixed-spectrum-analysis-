function sae = saetrain(sae, x, opts)
    for i = 1 : numel(sae.ae) %°´ÕÕ²ãÊıÑµÁ·
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
        sae.ae{i} = nntrain(sae.ae{i}, x, x, opts);
        t = nnff(sae.ae{i}, x, x);
        x = t.a{2};
        % remove bias term
        %x = x(:,1:end);
    end
end
