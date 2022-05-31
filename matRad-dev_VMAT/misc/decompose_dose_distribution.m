% recompute dij
run_dose_calc_gym

totalNumCP = max(size(resultGUI.apertureInfo.beam));
for i = 1:totalNumCP
    eval(['results.cp',num2str(i),' = compute_dose_contribution(',num2str(i),',dij,resultGUI,pln);'])
end