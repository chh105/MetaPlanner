function [outputMask] = volVecToMask(ct, volVec)
outputMask = zeros(ct.cubeDim);
outputMask(volVec)=1;

