function ant = antrenew(ant, opts)
%% Get Option value
nAnt = opts.nAnt;
new_nAnt = opts.new_nAnt;
nDim = opts.nDim;
prob = opts.prob;
eva_rate = opts.eva_rate;

%% Calculate the sigma
sigma = zeros(nAnt, size(ant(1).Position,2));
for i = 1:nAnt
        Dist = 0;
        for j = 1:nAnt
                Dist = Dist + abs(ant(i).Position-ant(j).Position);
        end
        sigma(i, :) = eva_rate .* Dist/(nAnt-1);
end

%% Roulette Wheel Selection to new Ant
ant(new_nAnt+nAnt).Position = zeros(1, size(ant(1).Position,2));
for i = (1:new_nAnt) + nAnt
        p = RouletteWheelSelection(prob);
        ant(i).Position = ant(p).Position + ant(p).Position .* randn(1, size(ant(1).Position,2)).*sigma(p, :);
end

end