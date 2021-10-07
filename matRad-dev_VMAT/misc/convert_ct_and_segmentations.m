num_structures = size(cst,1);
segmentation_matrix = zeros([ct.cubeDim,num_structures]);

for i = 1:num_structures
    
    structure_segmentation_matrix = zeros(ct.cubeDim);
    
    voxel_indices = cell2mat(cst{i,4});
    structure_segmentation_matrix(voxel_indices) = 1;
    segmentation_matrix(:,:,:,i) = structure_segmentation_matrix;
    
end

ct_cube_HU = ct.cubeHU{:};
structure_names = cst(:,2);


% output is segmentation_matrix, ct_cube_HU, and structure_names