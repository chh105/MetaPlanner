function resultGUI = compute_dose_contribution(cpToKeep,dij,resultGUI,pln)

totalNumCP = max(size(resultGUI.apertureInfo.beam));
for i=1:totalNumCP
    if i==cpToKeep
        continue
    else
        resultGUI.apertureInfo.apertureVector(i)=0;
    end
end

% set optimization options
options.radMod          = pln.radiationMode;
options.bioOpt          = pln.propOpt.bioOptimization;
options.ID              = [pln.radiationMode '_' pln.propOpt.bioOptimization];
options.FMO             = false; % let optimizer know that this is FMO
options.numOfScenarios  = dij.numOfScenarios;

% 
resultGUI.apertureInfo = matRad_daoVec2ApertureInfo(resultGUI.apertureInfo,resultGUI.apertureInfo.apertureVector);
resultGUI.w = resultGUI.apertureInfo.bixelWeights;
d = matRad_backProjection(resultGUI.w,dij,options);
resultGUI.physicalDose = reshape(d{1},dij.dimensions);