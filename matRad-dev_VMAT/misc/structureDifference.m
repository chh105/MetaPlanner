function [remainingVolVec] = structureDifference( ct, volVecA, volVecB )
% output remaining volume after A - B
% call
%   [remainingVolVec] = structureDifference( ct, volVecA, volVecB );
%
% input
%   ct:                     matRad ct object
%   volVecA:                volume A, check cst(:,4)
%   volVecB:                volume B, check cst(:,4)
%
% output
%   remainingVolVec:            remaining volume after A-B
%
%
dummy_vol_A = zeros(ct.cubeDim);
dummy_vol_A(volVecA)=1;
dummy_vol_B = zeros(ct.cubeDim);
dummy_vol_B(volVecB)=1;
dummy_vol_A_minus_B = dummy_vol_A - dummy_vol_B;

remainingVolVec = find(dummy_vol_A_minus_B>0);