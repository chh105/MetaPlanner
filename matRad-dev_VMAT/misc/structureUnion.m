function [unionVolVec] = structureUnion( ct, volVecA, volVecB )
% 
% call
%   [unionVolVec] = structureUnion( ct, volVecA, volVecB );
%
% input
%   ct:             matRad ct object
%   volVecA:                volume A, check cst(:,4)
%   volVecB:                volume B, check cst(:,4)
%
% output
%   unionVol:            union volume
%
%

dummy_vol_A = zeros(ct.cubeDim);
dummy_vol_A(volVecA)=1;
dummy_vol_B = zeros(ct.cubeDim);
dummy_vol_B(volVecB)=1;
dummy_vol_A_plus_B = dummy_vol_A + dummy_vol_B;

unionVolVec = find(dummy_vol_A_plus_B>0);

