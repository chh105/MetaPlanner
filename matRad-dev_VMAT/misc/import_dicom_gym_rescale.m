function [ct, cst, pln, stf] = import_dicom_gym_rescale( ctDir, rtStDir, mm_resolution, dicomMetaBool, useDoseGrid )
% matRad wrapper function to import a predefined set of dicom files into
% matRad's native data formats
% 
% call
%   [ct, cst, pln] = import_dicom_gym( ctDir, rtStDir, rtDoseDir, rtPlanDir, dicomMetaBool, useDoseGrid )
%
% input
%   ctDir:          path to directory containing ct dicoms
%   rtStDir:        path to directory containing rt structure dicoms
%   rtDoseDir:      (unused) path to directory containing rt dose dicoms
%   rtPlanDir:      (unused) path to directory containing rt plan dicoms
%   dicomMetaBool:  (boolean, optional) import complete dicomInfo and
%                   patientName
%   useDoseGrid:    (boolean, optional) use dose grid
%
% output
%   ct:        matRad ct struct
%   cst:       matRad cst struct
%   pln:       matRad plan struct
%   resultGUI: (unused) matRad result struct holding data for visualization in GUI
%
% References
%   -
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2015 the matRad development team. 
% 
% This file is part of the matRad project. It is subject to the license 
% terms in the LICENSE file found in the top-level directory of this 
% distribution and at https://github.com/e0404/matRad/LICENSES.txt. No part 
% of the matRad project, including this file, may be copied, modified, 
% propagated, or distributed except according to the terms contained in the 
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
if ~exist('rtDoseDir','var')
  useRtDose = false;
end
%
if ~exist('rtPlanDir','var')
  useRtPlan = false;
end
%
if ~exist('dicomMetaBool','var')
  dicomMetaBool = true;
end
%
if ~exist('useDoseGrid','var')
  useDoseGrid = false;
end
%
if ~exist('mm_resolution','var')
    mm_resolution = 5;
end


[ files.ct, ctPatientList ] = matRad_scanDicomImportFolder( ctDir );
[ files.rtss, rtStPatientList ] = matRad_scanDicomImportFolder( rtStDir );
if useRtDose
    [ files.rtdose, rtPlanPatientList ] = matRad_scanDicomImportFolder( rtDoseDir );
end
if useRtPlan
    [ files.rtplan, rtDosePatientList ] = matRad_scanDicomImportFolder( rtPlanDir );
end

files.resx = str2double(files.ct{1,9});
files.resy = str2double(files.ct{1,10});
files.resz = str2double(files.ct{1,11});
files.useDoseGrid = useDoseGrid;


% [ct, cst, pln, resultGUI] = matRad_importDicom(files, dicomMetaBool);
%% Script version of above function



[env, ~] = matRad_getEnvironment();
    

h = waitbar(0,'Please wait...');
%h.WindowStyle = 'Modal';
steps = 2;

%import ct-cube
waitbar(1 / steps)
resolution.x = mm_resolution;
resolution.y = mm_resolution;
resolution.z = mm_resolution; % [mm] / lps coordinate system
if files.useDoseGrid && isfield(files,'rtdose')
    % get grid from dose cube
    if verLessThan('matlab','9')
        doseInfo = dicominfo(files.rtdose{1,1});
    else
        doseInfo = dicominfo(files.rtdose{1,1},'UseDictionaryVR',true);
    end
    doseGrid{1} = doseInfo.ImagePositionPatient(1) + doseInfo.ImageOrientationPatient(1) * ...
                                                     doseInfo.PixelSpacing(1) * double(0:doseInfo.Columns - 1);
    doseGrid{2} = doseInfo.ImagePositionPatient(2) + doseInfo.ImageOrientationPatient(5) * ...
                                                     doseInfo.PixelSpacing(2) * double(0:doseInfo.Rows - 1);
    doseGrid{3} = doseInfo.ImagePositionPatient(3) + doseInfo.GridFrameOffsetVector(:)';

    % get ct on grid
    ct = matRad_importDicomCt(files.ct, resolution, dicomMetaBool,doseGrid); 

else
    ct = matRad_importDicomCt(files.ct, resolution, dicomMetaBool); 
end

if ~isempty(files.rtss)
    
    %% import structure data
    waitbar(2 / steps)
    structures = matRad_importDicomRtss(files.rtss{1},ct.dicomInfo);
    close(h)

    %% creating structure cube
    h = waitbar(0,'Please wait...');
    %h.WindowStyle = 'Modal';
    steps = numel(structures);
    for i = 1:numel(structures)
        % computations take place here
        waitbar(i / steps)
        fprintf('creating cube for %s volume...\n', structures(i).structName);
        structures(i).indices = matRad_convRtssContours2Indices(structures(i),ct);
    end
    fprintf('finished!\n');
    close(h)

    %% creating cst
    cst = matRad_createCst(structures);

else
    
    cst = matRad_dummyCst(ct);
    
end

% determine pln parameters
if isfield(files,'rtplan')
    if ~(cellfun(@isempty,files.rtplan(1,:)))
        pln = matRad_importDicomRTPlan(ct, files.rtplan, dicomMetaBool);
    end
end

% import stf
if isfield(files,'rtplan')
    if ~(cellfun(@isempty,files.rtplan(1,:)))
        if (strcmp(pln.radiationMode,'protons') || strcmp(pln.radiationMode,'carbon'))
            %% import steering file
            % pln output because bixelWidth is determined via the stf
            [stf, pln] = matRad_importDicomSteeringParticles(ct, pln, files.rtplan);
        elseif strcmp(pln.radiationMode, 'photons') && isfield(pln.propStf,'collimation')
            % return correct angles in pln 
            [stf, pln] = matRad_importDicomSteeringPhotons(pln);
        else
            warning('No support for DICOM import of steering information for this modality.');
        end
    end
else
    
    % default meta information for treatment plan

    pln.radiationMode   = 'photons';     % either photons / protons / carbon
    pln.machine         = 'Generic';

    pln.numOfFractions  = 40;

    % beam geometry settings
    pln.propStf.bixelWidth      = 10; % [mm] / also corresponds to lateral spot spacing for particles
%     pln.propStf.gantryAngles    = [0:72:359]; % [?]
%     pln.propStf.gantryAngles    = [0 30 60 150 180 210 300 330]; % [?]
%     pln.propStf.gantryAngles    = [0:360/8:359]; % [?]
%     pln.propStf.gantryAngles    = [20:360/11:359]; % [?] pops
    pln.propStf.gantryAngles    = [0:360/9:359]; % [?]
%     pln.propStf.gantryAngles    = [0,45,90,135,180,225,270,315]; % [?]
    pln.propStf.couchAngles     = zeros(1,9); % [?]
    pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
    pln.propStf.isoCenter       = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);

    % dose calculation settings
    pln.propDoseCalc.doseGrid.resolution.x = mm_resolution; % [mm]
    pln.propDoseCalc.doseGrid.resolution.y = mm_resolution; % [mm]
    pln.propDoseCalc.doseGrid.resolution.z = mm_resolution; % [mm]

    % optimization settings
    pln.propOpt.optimizer       = 'IPOPT';
    pln.propOpt.bioOptimization = 'none'; % none: physical optimization;             const_RBExD; constant RBE of 1.1;
                                          % LEMIV_effect: effect-based optimization; LEMIV_RBExD: optimization of RBE-weighted dose
    pln.propOpt.runDAO          = false;  % 1/true: run DAO, 0/false: don't / will be ignored for particles
    pln.propOpt.runSequencing   = false;  % 1/true: run sequencing, 0/false: don't / will be ignored for particles and also triggered by runDAO below
end
% 
% % import dose cube
% if isfield(files,'rtdose')
%     % check if files.rtdose contains a path and is labeld as RTDose
%     % only the first two elements are relevant for loading the rt dose
%     if ~(cellfun(@isempty,files.rtdose(1,1:2))) 
%         fprintf('loading Dose files \n', structures(i).structName);
%         % parse plan in order to scale dose cubes to a fraction based dose
%         if exist('pln','var')
%             if isfield(pln,'numOfFractions')
%                 resultGUI = matRad_importDicomRTDose(ct, files.rtdose, pln);
%             end
%         else
%             resultGUI = matRad_importDicomRTDose(ct, files.rtdose);
%         end
%         if size(resultGUI) == 0
%            clear resultGUI;
%         end
%     end
% end
% 
% % put weight also into resultGUI
% if exist('stf','var') && exist('resultGUI','var')
%     resultGUI.w = [];
%     for i = 1:size(stf,2)
%         resultGUI.w = [resultGUI.w; [stf(i).ray.weight]'];
%     end
% end

%% generate steering file
stf = matRad_generateStf(ct,cst,pln);