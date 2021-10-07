function [cst] = createHeadNeckTsStructures( ct, cst, target_idxs, target_pres_doses, target_exp_rs,  vox_size, body_idx )

if ~exist('target_exp_rs','var')
    target_exp_rs = [5]; % default expansion by 10 mm
end

if ~exist('vox_size','var')
    vox_size = 2; % default mm voxel size
end

if ~exist('body_idx','var')
    body_idx = nan; 
end

[cst] = createNonoverlappingPtvs( ct, cst, target_idxs, target_pres_doses );

if ~isnan(body_idx)
    [cst] = createRingStructure( ct, cst, target_idxs, target_exp_rs, vox_size, body_idx);
    [cst] = createAllTargetExpansion( ct, cst, target_idxs, target_exp_rs, vox_size, body_idx);
    cst = cst(:,1:6);
else
    [cst] = createRingStructure( ct, cst, target_idxs, target_exp_rs, vox_size);
    [cst] = createAllTargetExpansion( ct, cst, target_idxs, target_exp_rs, vox_size);
    cst = cst(:,1:6);
end

