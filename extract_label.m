position = 0
trial = 0

for i = drange(1:30)
    disp(position)
    disp(trial)
    filepath = sprintf('data/roi_dang/teeth/%d_%d.csv', position, trial)
    writetable(teeth.Labels.roi{i, 1}, filepath)

    if trial + 1 == 10
        trial = -1
        position = position + 1
    end

    trial = trial + 1
end

