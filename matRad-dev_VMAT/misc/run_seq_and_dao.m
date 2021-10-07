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