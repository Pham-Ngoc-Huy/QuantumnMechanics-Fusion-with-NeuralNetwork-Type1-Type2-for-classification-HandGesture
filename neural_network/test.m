% Add this RIGHT AFTER loading data, before anything else:
fprintf('NaN in all_data: %d\n', sum(isnan(all_data(:))));
fprintf('Inf in all_data: %d\n', sum(isinf(all_data(:))));