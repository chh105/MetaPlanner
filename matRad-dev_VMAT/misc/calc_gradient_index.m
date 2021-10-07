function qi = calc_gradient_index(cst,pln,doseCube,targetRefDose,param)

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



if ~exist('refVol', 'var') || isempty(refVol)
    refVol = [2 5 50 95 98];
end

if ~exist('refGy', 'var') || isempty(refGy)
    refGy = floor(linspace(0,max(doseCube(:)),6)*10)/10;
end

if exist('param','var')
   if ~isfield(param,'logLevel')
      param.logLevel = 1;
   end
else
   param.logLevel = 1;
end
    
% calculate QIs per VOI
qi = struct;
for runVoi = 1:size(cst,1)
    
    indices     = cst{runVoi,4}{1};
    numOfVoxels = numel(indices); 
    voiPrint = sprintf('%3d %20s',cst{runVoi,1},cst{runVoi,2}); %String that will print quality indicators
    
    % get Dose, dose is sorted to simplify calculations
    doseInVoi    = sort(doseCube(indices));
        
    if ~isempty(doseInVoi)
        
        qi(runVoi).name = cst{runVoi,2};
        
        % easy stats
        qi(runVoi).mean = mean(doseInVoi);
        qi(runVoi).std  = std(doseInVoi);
        qi(runVoi).max  = doseInVoi(end);
        qi(runVoi).min  = doseInVoi(1);

        voiPrint = sprintf('%s - Mean dose = %5.2f Gy +/- %5.2f Gy (Max dose = %5.2f Gy, Min dose = %5.2f Gy)\n%27s', ...
                           voiPrint,qi(runVoi).mean,qi(runVoi).std,qi(runVoi).max,qi(runVoi).min,' ');

        DX = @(x) matRad_interp1(linspace(0,1,numOfVoxels),doseInVoi,(100-x)*0.01);
        VX = @(x) numel(doseInVoi(doseInVoi >= x)) / numOfVoxels;

        % create VX and DX struct fieldnames at runtime and fill
        for runDX = 1:numel(refVol)
            qi(runVoi).(strcat('D_',num2str(refVol(runDX)))) = DX(refVol(runDX));
            voiPrint = sprintf('%sD%d%% = %5.2f Gy, ',voiPrint,refVol(runDX),DX(refVol(runDX)));
        end
        voiPrint = sprintf('%s\n%27s',voiPrint,' ');
        for runVX = 1:numel(refGy)
            sRefGy = num2str(refGy(runVX),3);
            qi(runVoi).(['V_' strrep(sRefGy,'.','_') 'Gy']) = VX(refGy(runVX));
            voiPrint = sprintf(['%sV' sRefGy 'Gy = %6.2f%%, '],voiPrint,VX(refGy(runVX))*100);
        end
        voiPrint = sprintf('%s\n%27s',voiPrint,' ');

        % if current voi is a target -> calculate homogeneity and conformity
        if strcmp(cst{runVoi,3},'TARGET') > 0      

            % loop over target objectives and get the lowest dose objective 
            referenceDose = targetRefDose;
            
            if isstruct(cst{runVoi,6})
                cst{runVoi,6} = num2cell(arrayfun(@matRad_DoseOptimizationFunction.convertOldOptimizationStruct,cst{runVoi,6}));
            end
            
            for runObjective = 1:numel(cst{runVoi,6})
               % check if this is an objective that penalizes underdosing 
               obj = cst{runVoi,6}{runObjective};
               if ~isa(obj,'matRad_DoseOptimizationFunction')
                   try
                       obj = matRad_DoseOptimizationFunction.createInstanceFromStruct(obj);
                   catch ME
                       warning('Objective/Constraint not valid!\n%s',ME.message)
                       continue;
                   end
               end
               
         
            end

            if referenceDose == targetRefDose 
                voiPrint = sprintf('%s%s',voiPrint,'Warning: target has no objective that penalizes underdosage, ');
            end

 
            StringReferenceDose = regexprep(num2str(round(referenceDose*100)/100),'\D','_');
            % Conformity Index, fieldname contains reference dose
            VTarget95 = sum(doseInVoi >= 0.95*referenceDose); % number of target voxels recieving dose >= 0.95 dPres
            VTreated95 = sum(doseCube(:) >= 95*referenceDose);  %number of all voxels recieving dose >= 0.95 dPres ("treated volume")
            VTreated50 = sum(doseCube(:) >= 0.50*referenceDose);  %number of all voxels recieving dose >= 0.50 dPres ("treated volume")
            VTreated90 = sum(doseCube(:) >= 0.9*referenceDose);  %number of all voxels recieving dose >= 0.90 dPres ("treated volume")
            qi(runVoi).(['GI']) = VTreated90/numOfVoxels; 

            voiPrint = sprintf('%sGI = %6.4f for reference dose of %3.2f Gy\n',voiPrint,...
                               qi(runVoi).(['GI']),referenceDose);
            
        end
        matRad_dispToConsole(voiPrint,param,'info','%s\n')
    
    else
        
        matRad_dispToConsole([num2str(cst{runVoi,1}) ' ' cst{runVoi,2} ' - No dose information.\n'],param,'info')
        
    end
end

% assign VOI names which could be corrupted due to empty structures
listOfFields = fieldnames(qi);
for i = 1:size(cst,1)
  indices     = cst{i,4}{1};
  doseInVoi    = sort(doseCube(indices));
  if isempty(doseInVoi)
      for j = 1:numel(listOfFields)
          qi(i).(listOfFields{j}) = NaN;
      end
      qi(i).name = cst{i,2};
  end
end

end

