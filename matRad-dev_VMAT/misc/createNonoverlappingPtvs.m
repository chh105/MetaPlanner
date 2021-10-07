function [cst] = createNonoverlappingPtvs( ct, cst, targetIdxs, targetPrescriptionDoses )
% call
%   [cst] = createNonoverlappingPtvs( ct, cst, targetIdxs, targetPrescriptionDoses );
%
% input
%   ct:                             matRad ct object
%   cst:                            matRad cst object
%   targetIdxs:                     vector of cst row indices of the 
%                                   target structures used
%   targetPrescriptionDoses:        vector of prescription doses for target
%                                   structures (ordered from low to high)
%
% output
%   cst:            matRad cst object
%
%

% sort in ascending order
[new_targetPrescriptionDoses, order_targetPrescriptionDoses] = sort(targetPrescriptionDoses);
new_targetIdxs = targetIdxs(order_targetPrescriptionDoses);

for i = 1:length(new_targetIdxs) - 1
    
    target_a = i;
    target_a_cst_idx = new_targetIdxs(target_a);
    target_a_prescription = new_targetPrescriptionDoses(target_a);
    remaining_a_vec = cst{target_a_cst_idx,4}{:};
    
    for j = 1:length(new_targetIdxs) - i
        
        target_b = length(new_targetIdxs) - j + 1;
        target_b_cst_idx = new_targetIdxs(target_b);
        target_b_prescription = new_targetPrescriptionDoses(target_b);
        
        b_dilation_radius = ceil((target_b_prescription - target_a_prescription)/3);
%         b_dilation_radius = 5;
        
        radius_mod_2 = mod(b_dilation_radius,2);
        b_dilation_radius = floor(b_dilation_radius/2) + radius_mod_2;

        exp_b_vec = dilateStructure(ct, cst, target_b_cst_idx, b_dilation_radius, 2);
        
        remaining_a_vec = structureDifference( ct, remaining_a_vec, exp_b_vec );
    end
    
    cst(end+1,:) = cst(target_a_cst_idx,:);
    cst(end,2) = {['NO_',cst{target_a_cst_idx,2}]};
    cst(end,6) = {[]};
    cst{end,4}{:} = remaining_a_vec;
    
    
end

cst = cst(:,1:6);