clear;
clc;

base_path = "D:\system_indentification_with_NN_fuzzy\data\DB9";

all_mat = dir(fullfile(base_path, 's*_angles', 's*_angles', '*.mat'));

angles = {};
exercise = {};
glove = {};
rerepetition = {};
restimulus = {};
stimulus = {};
repetition = {};
subject = {};
for i = 1:length(all_mat)
    file_path = fullfile(all_mat(i).folder, all_mat(i).name);
    data = load(file_path);

    N = size(data.angles,1);
    angles{end+1} = data.angles;
    glove{end+1} = data.glove;
    rerepetition{end+1} = data.rerepetition;
    restimulus{end+1} = data.restimulus;
    stimulus{end+1}= data.stimulus;
    repetition{end+1}= data.repetition;

    subject{end+1} = repmat(data.subject, N, 1);
    exercise{end+1} = repmat(data.exercise, N, 1);
    fprintf("Loaded: %s\n", file_path);
end

data_angles = vertcat(angles{:});
data_exercises = vertcat(exercise{:});
data_rerepetition = vertcat(rerepetition{:});
data_restimulus = vertcat(restimulus{:});
data_stimulus = vertcat(stimulus{:});
data_repetition = vertcat(repetition{:});
data_subject = vertcat(subject{:});

all_data = [data_subject,data_exercises, data_angles,...
    data_rerepetition(1:2255046, :),data_restimulus,...
    data_stimulus(1:2255046, :)];

fprintf('all_data_table: %d rows x %d cols\n', size(all_data,1), size(all_data,2));
save("processed_data_movement_classification.mat", "all_data")
