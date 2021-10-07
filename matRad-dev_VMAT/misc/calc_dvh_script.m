if isfield(resultGUI,'RBExDose')
    doseCube = resultGUI.RBExDose;
else
    doseCube = resultGUI.physicalDose;
end

if ~exist('refVol', 'var') 
    refVol = [];
end

if ~exist('refGy', 'var')
    refGy = [];
end

if exist('param','var')
   if ~isfield(param,'logLevel')
      param.logLevel = 1;
   end
else
   param.logLevel = 1;
end

dvh = matRad_calcDVH(cst,doseCube,'cum');