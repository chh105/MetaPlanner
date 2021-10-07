
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