% Load saved data
load('./optoforce_raw_coords_1.mat');

% Constant windowing parameters
threshold = 3; % Threshold value for bump detection
window_size = 50; % Window size for detection

% Initialize variables to store detected bump indices
bump_indices = [];

% Iterate through data to detect bumps
i = window_size + 1; % Start from end of window
while i <= length(res_t)
    % Calculate mean within window
    window_mean = mean(res_t(i - window_size:i));
    % If current value is non-zero and exceeds threshold, it's the start of a bump
    if res_t(i) > threshold && res_t(i) > window_mean && res_t(i) ~= 0
        % Find local maximum
        [~, max_index] = max(res_t(i:i + window_size));
        % Find local minimum after local maximum
        [~, min_index] = min(res_t(i + max_index:i + window_size));
        % Ensure minimum index is after maximum index
        if min_index > max_index
            % Add indices of bump start and associated trough
            bump_indices(end + 1) = i + max_index - 1;
            bump_indices(end + 1) = i + max_index + min_index - 1;
            % Move to index after local minimum
            i = i + max_index + min_index;
        else
            % Move to next index
            i = i + 1;
        end
    else
        % Move to next index
        i = i + 1;
    end
end

% Plot data with detected bumps
clf
figure(1);
plot(res_t, 'b*-');
hold on;
plot(bump_indices, res_t(bump_indices), 'ro', 'MarkerSize', 10);
% Add red area between local maxima and minima
for k = 1:2:length(bump_indices)
    if k < length(bump_indices)
        max_index = bump_indices(k);
        min_index = bump_indices(k+1);
        % Plot red line between local maximum and minimum
        plot(max_index:min_index, res_t(max_index:min_index), 'r', 'LineWidth', 2);
    end
end
xlabel('Sample');
ylabel('Resultant Force');
title('Bump Detection');
legend('Resultant Force', 'Detected Bumps', 'Area between local max and min', 'Location', 'best');
grid on;

% Display detected bump indices
disp('Detected Bump Indices:');
disp(bump_indices);
