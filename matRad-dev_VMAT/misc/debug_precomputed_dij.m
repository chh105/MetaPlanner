gantry = [90 -90];
couch = [0 0];
pln.propStf.gantryAngles = gantry;
pln.propStf.couchAngles = couch;
pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);
pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);
stf = matRad_generateStf(ct,cst,pln);

% compute dij for first angle
pln.propStf.gantryAngles = gantry(1);
pln.propStf.couchAngles = couch(1);
pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);
stf = matRad_generateStf(ct,cst,pln);
if strcmp(pln.radiationMode,'photons')
    dij_1 = matRad_calcPhotonDose(ct,stf,pln,cst);
    %dij = matRad_calcPhotonDoseVmc(ct,stf,pln,cst);
elseif strcmp(pln.radiationMode,'protons') || strcmp(pln.radiationMode,'carbon')
    dij_1 = matRad_calcParticleDose(ct,stf,pln,cst);
end

% compute dij for second angle
pln.propStf.gantryAngles = gantry(2);
pln.propStf.couchAngles = couch(2);
pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);
stf = matRad_generateStf(ct,cst,pln);
if strcmp(pln.radiationMode,'photons')
    dij_2 = matRad_calcPhotonDose(ct,stf,pln,cst);
    %dij = matRad_calcPhotonDoseVmc(ct,stf,pln,cst);
elseif strcmp(pln.radiationMode,'protons') || strcmp(pln.radiationMode,'carbon')
    dij_2 = matRad_calcParticleDose(ct,stf,pln,cst);
end

% compute dij for 2 beam config
pln.propStf.gantryAngles = gantry;
pln.propStf.couchAngles = couch;
pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);
stf = matRad_generateStf(ct,cst,pln);
if strcmp(pln.radiationMode,'photons')
    dij = matRad_calcPhotonDose(ct,stf,pln,cst);
    %dij = matRad_calcPhotonDoseVmc(ct,stf,pln,cst);
elseif strcmp(pln.radiationMode,'protons') || strcmp(pln.radiationMode,'carbon')
    dij = matRad_calcParticleDose(ct,stf,pln,cst);
end

% precomputed dose mat
dose_mat = [];
dose_mat = [dose_mat,dij_1.physicalDose{1}];
dose_mat = [dose_mat,dij_2.physicalDose{1}];
% regular dose mat
regular_dose_mat = dij.physicalDose{1};

%% comparisons
subplot(2,1,1)
spy(dose_mat)
subplot(2,1,2)
spy(regular_dose_mat)
sprintf('NNZ precomputed:%d',nnz(dose_mat))
sprintf('NNZ regular:%d',nnz(regular_dose_mat))
diff = full(sum(dose_mat(:)-regular_dose_mat(:))^2/sum(dose_mat(:))^2);
sprintf('Diff:%e',diff)



