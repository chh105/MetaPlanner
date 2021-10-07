function [exp_dummy_vol_vec] = dilateStructure( ct, cst, cstRowIdx, r, vox_size, up_scale )
% 
% call
%   [exp_dummy_vol_vec] = dilateStructure(ct, cst, cstRowIdx, r, up_scale);
%
% input
%   ct:             matRad ct object
%   cst:            matRad cst object
%   cstRowIdx:      cst row index of the structure used for dilation
%   r:              radius for dilating structure
%   up_scale:       high resolution cubeDim 
%
% output
%   exp_dummy_vol_vec:            dilated volume
%
%
if ~exist('r','var')
    r = 6; % default expansion by 6 mm
end
if ~exist('vox_size','var')
    vox_size = 1; % default mm voxel size
end
if ~exist('up_scale','var')
    rows = ct.resolution.x*ct.cubeDim(1);
    cols = ct.resolution.y*ct.cubeDim(2);
    planes = ct.resolution.z*ct.cubeDim(3);
    up_scale = [rows,cols,planes]./vox_size;
end


dummy_vol = zeros(ct.cubeDim);
dummy_vol(cst{cstRowIdx,4}{:})=1;

if r > 0
    dummy_vol_high_res = imresize3(dummy_vol,up_scale,'nearest');
    exp_dummy_vol_high_res = imdilate(dummy_vol_high_res,strel('sphere',r));
    down_scale = ct.cubeDim;
    exp_dummy_vol = imresize3(exp_dummy_vol_high_res,down_scale,'nearest');
    exp_dummy_vol_vec = find(exp_dummy_vol>0);
else
    exp_dummy_vol_vec = cst{cstRowIdx,4}{:};
end

