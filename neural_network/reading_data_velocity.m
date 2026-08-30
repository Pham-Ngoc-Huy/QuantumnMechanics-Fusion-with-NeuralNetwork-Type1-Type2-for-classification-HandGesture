clear; clc;
base_path = "D:\system_indentification_with_NN_fuzzy\data\data\MAT files";
listing = dir(base_path);
listing = listing([listing.isdir]);
listing = listing(~ismember({listing.name}, {'.', '..'}));

%% scaning columns in each cell of each leg
all_emg_cols    = struct();
all_torque_cols = struct();

for p = 1:length(listing)
    matPath = fullfile(base_path, listing(p).name, "Processed_Data.mat");
    if ~isfile(matPath), continue; end
    data = load(matPath);
    subjectField = fieldnames(data);
    subject = data.(subjectField{1});
    V_fields = fieldnames(subject);
    for v = 1:length(V_fields)
        Vdata = subject.(V_fields{v});
        sides = fieldnames(Vdata);
        for s = 1:length(sides)
            side_name  = sides{s};
            emg_raw    = Vdata.(side_name).EMGs_filt;
            torque_raw = Vdata.(side_name).Torques_Norm;
            if ~isempty(emg_raw)
                for k = 1:size(emg_raw, 1)
                    tbl = emg_raw{k, 2};
                    if ~istable(tbl) || isempty(tbl), continue; end
                    if ~isfield(all_emg_cols, side_name), all_emg_cols.(side_name) = {}; end
                    all_emg_cols.(side_name) = union(all_emg_cols.(side_name), tbl.Properties.VariableNames);
                end
            end
            if ~isempty(torque_raw)
                for k = 1:size(torque_raw, 1)
                    tbl = torque_raw{k, 2};
                    if ~istable(tbl) || isempty(tbl), continue; end
                    if ~isfield(all_torque_cols, side_name), all_torque_cols.(side_name) = {}; end
                    all_torque_cols.(side_name) = union(all_torque_cols.(side_name), tbl.Properties.VariableNames);
                end
            end
        end
    end
end

% check the result of right/left leg
sides_found = fieldnames(all_emg_cols);
for s = 1:length(sides_found)
    fprintf('EMG_%s (%d cols): %s\n', sides_found{s}, numel(all_emg_cols.(sides_found{s})), strjoin(all_emg_cols.(sides_found{s}), ', '));
end
sides_found = fieldnames(all_torque_cols);
for s = 1:length(sides_found)
    fprintf('Torque_%s (%d cols): %s\n', sides_found{s}, numel(all_torque_cols.(sides_found{s})), strjoin(all_torque_cols.(sides_found{s}), ', '));
end

%% start processing
all_emg_R = {}; all_torque_R = {};
all_emg_L = {}; all_torque_L = {};
all_participant = {};
all_velocity    = {};
all_side        = {};

for p = 1:length(listing)
    matPath = fullfile(base_path, listing(p).name, "Processed_Data.mat");
    if ~isfile(matPath), continue; end
    data = load(matPath);
    subjectField = fieldnames(data);
    subject = data.(subjectField{1});
    V_fields = fieldnames(subject);

    for v = 1:length(V_fields)
        Vdata = subject.(V_fields{v});
        vel_num = str2double(regexp(V_fields{v}, '\d+', 'match', 'once'));
        sides = fieldnames(Vdata);

        for s = 1:length(sides)
            side_name  = sides{s};
            emg_raw    = Vdata.(side_name).EMGs_filt;
            torque_raw = Vdata.(side_name).Torques_Norm;
            if isempty(emg_raw), continue; end

            master_emg    = all_emg_cols.(side_name);
            master_torque = all_torque_cols.(side_name);

            for k = 1:size(emg_raw, 1)
                tbl_emg = emg_raw{k, 2};
                if ~istable(tbl_emg) || isempty(tbl_emg), continue; end
                n_rows = size(tbl_emg, 1);
                % processing for EMG
                mat_emg = zeros(n_rows, numel(master_emg));
                for c = 1:numel(tbl_emg.Properties.VariableNames)
                    col_name = tbl_emg.Properties.VariableNames{c};
                    idx = strcmp(master_emg, col_name);
                    mat_emg(:, idx) = tbl_emg{:, c};
                end
                % processing for Torque Norm
                mat_torque = zeros(n_rows, numel(master_torque));
                tbl_torque = torque_raw{k, 2};
                if istable(tbl_torque) && ~isempty(tbl_torque)
                    for c = 1:numel(tbl_torque.Properties.VariableNames)
                        col_name = tbl_torque.Properties.VariableNames{c};
                        idx = strcmp(master_torque, col_name);
                        mat_torque(:, idx) = tbl_torque{:, c};
                    end
                end

                if contains(side_name, 'R')
                    all_emg_R{end+1}    = mat_emg;
                    all_torque_R{end+1} = mat_torque;
                    % need record one time only
                    all_participant{end+1} = repmat(p, n_rows, 1);
                    all_velocity{end+1}    = repmat(vel_num, n_rows, 1);
                else % this is for left leg
                    all_emg_L{end+1}    = mat_emg;
                    all_torque_L{end+1} = mat_torque;
                end


            end
        end
    end
end

% converting from cell to double
data_emg_R       = vertcat(all_emg_R{:});
data_torque_R    = vertcat(all_torque_R{:});
data_emg_L       = vertcat(all_emg_L{:});
data_torque_L    = vertcat(all_torque_L{:});
label_participant = vertcat(all_participant{:});
label_velocity   = vertcat(all_velocity{:});

% mergeing into one -> one table only
all_data = [label_participant, label_velocity, data_emg_R, data_emg_L, data_torque_R, data_torque_L];

% result size
fprintf('Right - EMG: %dx%d | Torque: %dx%d\n', size(data_emg_R,1), size(data_emg_R,2), size(data_torque_R,1), size(data_torque_R,2));
fprintf('Left  - EMG: %dx%d | Torque: %dx%d\n', size(data_emg_L,1), size(data_emg_L,2), size(data_torque_L,1), size(data_torque_L,2));
fprintf('Participants: %d unique | MatrixSize %dx%d\n', numel(unique(label_participant)), size(label_participant,1),size(label_participant,2));
fprintf('Velocities:   %d unique | MatrixSize %dx%d\n', numel(unique(label_velocity)), size(label_velocity,1),size(label_participant,2));

%% load in metadata
meta_table = [];

for p = 1:length(listing)
    txtPath = fullfile(base_path, listing(p).name, "Metadata.txt");
    if ~isfile(txtPath), continue; end

    info_str = fileread(txtPath);

    subject_id  = str2double(regexp(info_str, 'Subject(\d+)',            'tokens', 'once'));
    gender_str  = regexp(info_str, 'Gender:\s*(\w+)', 'tokens', 'once');

    if strcmpi(gender_str, 'Male')
        gender_decode = 1;
    else
        gender_decode = 2;
    end
    age         = str2double(regexp(info_str, 'Age: (\d+)',              'tokens', 'once'));
    body_height = str2double(regexp(info_str, 'Body Height: ([\d.]+)',   'tokens', 'once'));
    body_mass   = str2double(regexp(info_str, 'Body Mass: ([\d.]+)',     'tokens', 'once'));
    leg_length  = str2double(regexp(info_str, 'Leg Length: ([\d.]+)',    'tokens', 'once'));
    foot_length = str2double(regexp(info_str, 'Foot Length: ([\d.]+)',   'tokens', 'once'));

    meta_table = [meta_table; p, gender_decode, age, body_height, body_mass, leg_length, foot_length];
end

[~, loc] = ismember(all_data(:,1), meta_table(:,1));
if any(loc == 0)
    error('Some participants not found in metadata');
end
meta_expanded = meta_table(loc, 2:end);
all_data_full = [all_data, meta_expanded];
fprintf('all_data_table: %d rows x %d cols\n', size(all_data_full,1), size(all_data_full,2));
save("processed_data_velocity_classification.mat", "all_data_full")
%% convert to table with names
% meta_table = array2table(meta_table, 'VariableNames', {'participant', 'subject_id', 'age', 'body_height', 'body_mass', 'leg_length', 'foot_length'});
%
% all_data_table = array2table(all_data, 'VariableNames', {'participant', 'velocity', ...
%
%     'St1_VL_R', 'St2_VL_R', ...
%     'St1_BF_R', 'St2_BF_R', ...
%     'St1_TA_R', 'St2_TA_R', ...
%     'St1_GAL_R', 'St2_GAL_R', ...
%
%     'St1_VL_L', 'St1_BF_L', 'St1_TA_L', 'St1_GAL_L', ...
%
%     'St1_Pelvis_X_R', 'St1_Pelvis_Y_R', 'St1_Pelvis_Z_R', ...
%     'St2_Pelvis_X_R', 'St2_Pelvis_Y_R', 'St2_Pelvis_Z_R', ...
%
%     'St1_Hip_X_R', 'St1_Hip_Y_R', 'St1_Hip_Z_R', ...
%     'St2_Hip_X_R', 'St2_Hip_Y_R', 'St2_Hip_Z_R', ...
%
%     'St1_Knee_X_R', 'St1_Knee_Y_R', 'St1_Knee_Z_R', ...
%     'St2_Knee_X_R', 'St2_Knee_Y_R', 'St2_Knee_Z_R', ...
%
%     'St1_Ankle_X_R', 'St1_Ankle_Y_R', 'St1_Ankle_Z_R', ...
%     'St2_Ankle_X_R', 'St2_Ankle_Y_R', 'St2_Ankle_Z_R', ...
%
%     'St1_Hip_X_L', 'St1_Hip_Y_L', 'St1_Hip_Z_L', ...
%     'St1_Knee_X_L', 'St1_Knee_Y_L', 'St1_Knee_Z_L', ...
%     'St1_Ankle_X_L', 'St1_Ankle_Y_L', 'St1_Ankle_Z_L'
% });
