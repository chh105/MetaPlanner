function [cst] = createAllTargetExpansion( ct, cst, target_idxs, target_exp_rs,  vox_size, body_idx )

if ~exist('target_exp_rs','var')
    target_exp_rs = [5]; % default expansion by 10 mm
end

if ~exist('vox_size','var')
    vox_size = 2; % default mm voxel size
end

if ~exist('body_idx','var')
    body_idx = nan; 
end


rows = ct.resolution.x*ct.cubeDim(1);
cols = ct.resolution.y*ct.cubeDim(2);
planes = ct.resolution.z*ct.cubeDim(3);
up_scale = [rows,cols,planes]./vox_size;

%%
cst(end+1,:) = cst(target_idxs(1),:);
cst(end,2) = {'mrad_all_target_expansion'};
cst(end,3) = {'TARGET'};
cst(end,6) = {[]};
cst{end,5}.Priority = cst{target_idxs(1),5}.Priority+1;


target_exp_vol_vec = dilateStructure( ct, cst, target_idxs(1), target_exp_rs(1), up_scale );

for i = 2:length(target_idxs)
    i_target_exp_vol_vec = dilateStructure( ct, cst, target_idxs(i), target_exp_rs(i), up_scale );
    [target_exp_vol_vec] = structureUnion( ct, target_exp_vol_vec, i_target_exp_vol_vec );
    
end

if ~isnan(body_idx)
    [intersectVolVec] = structureIntersection( ct, target_exp_vol_vec, cst{body_idx,4}{:} );
    cst{end,4}{:} = intersectVolVec;
    
else
    cst{end,4}{:} = target_exp_vol_vec;
end

for i = 1:size(cst,1)
    if strcmp(cst{i,3},'OAR')
        orig_vol = cst{i,4}{:};
        [orig_sub_all_target] = structureDifference( ct, orig_vol, cst{end,4}{:} );
        cst{i,4}{:} = orig_sub_all_target;
    end
end
