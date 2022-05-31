function result = matRad_calcDeliveryTime(result, angle_threshold)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% matRad delivery metric calculation
%
% call
%   matRad_calcDeliveryTime(result)
%
% input
%   result:             result struct from fluence optimization/sequencing
%
% output
%   VMAT plans: total time, 
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2016 the matRad development team.
%
% This file is part of the matRad project. It is subject to the license
% terms in the LICENSE file found in the top-level directory of this
% distribution and at https://github.com/e0404/matRad/LICENSES.txt. No part
% of the matRad project, including this file, may be copied, modified,
% propagated, or distributed except according to the terms contained in the
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

apertureInfo = result.apertureInfo;
if nargin < 2
    angle_threshold = apertureInfo.beam(2).gantryAngle-apertureInfo.beam(1).gantryAngle;
end

fileName = apertureInfo.propVMAT.machineConstraintFile;
try
    load(fileName,'machine');
catch
    error(['Could not find the following machine file: ' fileName ]);
end
rot_times = zeros(1,size(apertureInfo.beam,2)-1);
for i=1:size(apertureInfo.beam,2)
    if i+1 > size(apertureInfo.beam,2)
        standard_dist = apertureInfo.beam(i).gantryAngle-apertureInfo.beam(i-1).gantryAngle;
    else
        standard_dist = apertureInfo.beam(i+1).gantryAngle-apertureInfo.beam(i).gantryAngle;
    end

    if standard_dist > angle_threshold
%         disp('MU Rate = 0')
        rot_speed = machine.constraints.gantryRotationSpeed(2);
    else
        rot_speed = machine.constraints.gantryRotationSpeed(2)*apertureInfo.beam(i).gantryRot;
    end
    
    rot_times(i) = standard_dist/rot_speed/10;
end
apertureInfo.planTime = sum(rot_times(:));
apertureInfo.rotTimes = rot_times;
result.apertureInfo = apertureInfo;

