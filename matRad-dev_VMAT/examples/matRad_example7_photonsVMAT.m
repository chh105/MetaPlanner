%% For VMAT

clear
close all
clc

% load patient data, i.e. ct, voi, cst

%load HEAD_AND_NECK
% load TG119.mat
load PROSTATE.mat
%load LIVER.mat
%load BOXPHANTOM.mat

% meta information for treatment plan

pln.radiationMode   = 'photons';   % either photons / protons / carbon
pln.machine         = 'Generic';

pln.numOfFractions  = 40;

% beam geometry settings
pln.propStf.bixelWidth = 7;

% dose calculation settings
pln.propDoseCalc.vmc                        = false;
% pln.propDoseCalc.vmcOptions.source          = 'phsp';
% pln.propDoseCalc.vmcOptions.phspBaseName    = '5x5_at_50cm';
% pln.propDoseCalc.vmcOptions.SCD             = 500;
% pln.propDoseCalc.vmcOptions.dumpDose        = 1;
% pln.propDoseCalc.vmcOptions.version         = 'Carleton';
% pln.propDoseCalc.vmcOptions.nCasePerBixel   = 5000;
% pln.propDoseCalc.vmcOptions.numOfParMCSim   = 8;


% optimization settings
pln.propOpt.bioOptimization = 'none';
pln.propOpt.runVMAT = true;
pln.propOpt.runDAO = true;
pln.propOpt.runSequencing = true;
pln.propOpt.preconditioner = true;
pln.propOpt.numLevels = 7;

pln.propOpt.VMAToptions.machineConstraintFile = [pln.radiationMode '_' pln.machine];
pln.propOpt.VMAToptions.continuousAperture = true;

pln.propOpt.VMAToptions.startingAngle = -180;
pln.propOpt.VMAToptions.finishingAngle = 180;
pln.propOpt.VMAToptions.maxGantryAngleSpacing = 4;      % Max gantry angle spacing for dose calculation
pln.propOpt.VMAToptions.maxDAOGantryAngleSpacing = 4;      % Max gantry angle spacing for DAO
pln.propOpt.VMAToptions.maxFMOGantryAngleSpacing = 28;      % Max gantry angle spacing for FMO

pln = matRad_VMATGantryAngles(pln,cst,ct);

%% initial visualization and change objective function settings if desired
matRadGUI

%% generate steering file
stf = matRad_generateStf(ct,cst,pln);

%% dose calculation
if strcmp(pln.radiationMode,'photons')
    if pln.propDoseCalc.vmc
        dij = matRad_calcPhotonDoseVmc(ct,stf,pln,cst);
    else
        dij = matRad_calcPhotonDose(ct,stf,pln,cst);
    end
elseif strcmp(pln.radiationMode,'protons') || strcmp(pln.radiationMode,'carbon')
    dij = matRad_calcParticleDose(ct,stf,pln,cst);
end
%dij.weightToMU = 100*(100/90)^2*(67/86)*(110/105)^2*(90/95)^2;

%this is equal to multiplication of factors:
% - factor when reference conditions are equal to each other (100)
% - inverse square factor to get same SSD
% - PDD factor (evaluated at SSD = 100 cm) (Podgorsak IAEA pg. 183)
% - Mayneord factor to move SSD from 100 cm to 85 cm

%At TOH: 100 cm SAD, 5 cm depth, 10x10cm2
%At DKFZ: 104 cm SAD, 5 cm depth, 5x5cm2

%% inverse planning for imrt
resultGUI = matRad_fluenceOptimization(dij,cst,pln,stf);

%% sequencing
if strcmp(pln.radiationMode,'photons') && (pln.propOpt.runSequencing || pln.propOpt.runDAO)
    %resultGUI = matRad_xiaLeafSequencing(resultGUI,stf,dij,5);
    %resultGUI = matRad_engelLeafSequencing(resultGUI,stf,dij,5);
    resultGUI = matRad_siochiLeafSequencing(resultGUI,stf,dij,pln,0);
    %resultGUI = matRad_svenssonLeafSequencing(resultGUI,stf,dij,pln,0);
end

%% DAO
if strcmp(pln.radiationMode,'photons') && pln.propOpt.runDAO
   resultGUI = matRad_directApertureOptimization(dij,cst,resultGUI.apertureInfo,resultGUI,pln,stf);
%    matRad_visApertureInfo(resultGUI.apertureInfo);
end

%% start gui for visualization of result
matRadGUI

%% indicator calculation and show DVH and QI
[dvh,qi] = matRad_indicatorWrapper(cst,pln,resultGUI);
