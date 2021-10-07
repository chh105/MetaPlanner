function [cst] = createNonoverlappingOar( ct, cst, oarIdx, targetIdxs )
% creates oars that have no overlap with targets
% call
%   [cst] = createNonoverlappingOar( ct, cst, oarIdx, targetIdxs );
%
% input
%   ct:                             matRad ct object
%   cst:                            matRad cst object
%   oarIdx:                         cst row index of the 
%                                   oar structure used
%   targetIdxs:                     vector of cst row indices of the 
%                                   target structures used
%
% output
%   cst:            matRad cst object
%
%


cst(end+1,:) = cst(oarIdx,:);
cst(end,2) = {['NO_',cst{oarIdx,2}]};
cst(end,6) = {[]};

for i = 1:length(targetIdxs)
    [remainingVolVec] = structureDifference( ct, cst{end,4}{:}, cst{targetIdxs(i),4}{:} );
    cst{end,4}{:} = remainingVolVec;
end
cst = cst(:,1:6);