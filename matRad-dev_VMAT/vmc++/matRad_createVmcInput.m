function matRad_createVmcInput(VmcOptions,filename)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% matRad vmc++ inputfile creation
%
% call
%   matRad_createVmcInput(VmcOptions,filename)
%
% input
%   VmcOptions:     structure set with VMC options
%   filename:       full file name of generated vmc input file (has to be 
%                   located in the runs path in the vmc++ folder)
%
%
% References
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% define a cell array which is bigger than necessary
% some parts of the input (e.g., the source) are variable
% then delete the empty elements
VmcInput = cell(100,1);
offset = 0;

% define the scoring options
VmcInput(offset+(1:11)) = {...
    ['  :start scoring options:     '                                                           ]   ,...
    ['      start in geometry: ' VmcOptions.scoringOptions.startInGeometry                      ]   ,...
    ['      :start dose options:    '                                                           ]   ,...
    ['          score in geometries: ' VmcOptions.scoringOptions.doseOptions.scoreInGeometries  ]   ,...
    ['          score dose to water: ' VmcOptions.scoringOptions.doseOptions.scoreDoseToWater   ]   ,...
    ['      :stop dose options:     '                                                           ]   ,...
    ['      :start output options ' VmcOptions.scoringOptions.outputOptions.name ':'            ]   ,...
    ['          dump dose: ' num2str(VmcOptions.scoringOptions.outputOptions.dumpDose)          ]   ,...
    ['      :stop output options ' VmcOptions.scoringOptions.outputOptions.name  ':'            ]   ,...
    ['  :stop scoring options:      '                                                           ]   ,...
    ['                              '                                                           ]   ...
};
offset = offset+11;

% define the geometry
VmcInput(offset+(1:8)) = {...
    ['  :start geometry:            '                                               ]   ,...
    ['      :start XYZ geometry:    '                                               ]   ,...
    ['          my name = ' VmcOptions.geometry.XyzGeometry.Ct                      ]   ,...
    ['          method of input = ' VmcOptions.geometry.XyzGeometry.methodOfInput   ]   ,...
    ['          phantom file    = ' VmcOptions.geometry.XyzGeometry.CtFile          ]   ,...
    ['      :stop XYZ geometry:     '                                               ]   ,...
    ['  :stop geometry:             '                                               ]   ,...
    ['                              '                                               ]   ...
};
offset = offset+8;

% define the source
if strcmp(VmcOptions.source.type,'beamlet')
    
    VmcInput(offset+(1:3)) = {...
        ['  :start beamlet source:      '                                                               ]   ,...
        ['      my name = ' VmcOptions.source.myName                                                    ]   ,...
        ['      monitor units ' VmcOptions.source.myName ' = ' num2str(VmcOptions.source.monitorUnits)  ]   ...
        };
    offset = offset+3;
    
    if ~isempty(VmcOptions.source.monoEnergy) && VmcOptions.source.monoEnergy>0
        VmcInput(offset+1) = {...
            ['      mono energy = ' num2str(VmcOptions.source.monoEnergy)   ]   ...
            };
        offset = offset+1;
    end
    
    VmcInput(offset+(1:6)) = {...
        ['      spectrum = ' VmcOptions.source.spectrum                                                             ]   ,...
        ['      charge       = ' num2str(VmcOptions.source.charge)                                                  ]   ,...
        ['      beamlet edges = ' num2str(VmcOptions.source.beamletEdges, '%8.5f ')                                 ]   ,...
        ['      virtual point source position = ' num2str(VmcOptions.source.virtualPointSourcePosition, '%8.5f ')   ]   ,...
        ['  :stop beamlet source:       '                                                                           ]   ,...
        ['                              '                                                                           ]   ...
        };
    offset = offset+6;
    
elseif strcmp(VmcOptions.source.type,'phsp')
    
    VmcInput(offset+(1:12)) = {...
        ['  :start general source:      '                                                               ]   ,...
        ['      monitor units ' VmcOptions.source.myName ' = ' num2str(VmcOptions.source.monitorUnits)  ]   ,...
        ['      translation ' VmcOptions.source.myName ' = ' num2str(VmcOptions.source.translation)     ]   ,...
        ['      isocenter ' VmcOptions.source.myName ' = ' num2str(VmcOptions.source.isocenter)         ]   ,...
        ['      angles ' VmcOptions.source.myName ' = ' num2str(VmcOptions.source.angles)               ]   ,...
        ['      :start phsp source:     '                                                               ]   ,...
        ['          my name = ' VmcOptions.source.myName                                                ]   ,...
        ['          file name = ' VmcOptions.source.file_name                                           ]   ,...
        ['          particle type = ' num2str(VmcOptions.source.particleType)                           ]   ,...
        ['      :stop phsp source:      '                                                               ]   ,...
        ['  :stop general source:       '                                                               ]   ,...
        ['                              '                                                               ]   ...
    };
offset = offset+12;
end

% define the MC parameters
VmcInput(offset+(1:5)) = {...
    ['  :start MC Parameter:        '                                           ]   ,...
    ['      automatic parameter = ' VmcOptions.McParameter.automatic_parameter  ]   ,...
    ['      spin = ' num2str(VmcOptions.McParameter.spin)                       ]   ,...
    ['  :stop MC Parameter:         '                                           ]   ,...
    ['                              '                                           ]   ...
    };
offset = offset+5;

% define the MC control
VmcInput(offset+(1:6)) = {...
    ['  :start MC Control:          '                               ]   ,...
    ['      ncase  = ' num2str(VmcOptions.McControl.ncase)          ]   ,...
    ['      nbatch = ' num2str(VmcOptions.McControl.nbatch)         ]   ,...
    ['      rng seeds = ' num2str(VmcOptions.McControl.rngSeeds)    ]   ,...
    ['  :stop MC Control:           '                               ]   ,...
    ['                              '                               ]   ...
};
offset = offset+6;

% define the variance reduction
VmcInput(offset+(1:6)) = {...
    ['  :start variance reduction:  '                                                       ]   ,...
    ['      repeat history   = ' num2str(VmcOptions.varianceReduction.repeatHistory)        ]   ,...
    ['      split photons = ' num2str(VmcOptions.varianceReduction.splitPhotons)            ]   ,...
    ['      photon split factor = ' num2str(VmcOptions.varianceReduction.photonSplitFactor) ]   ,...
    ['  :stop variance reduction:   '                                                       ]   ,...
    ['                              '                                                       ]   ...
};
offset = offset+6;

% define the quasi
VmcInput(offset+(1:5)) = {...
    ['  :start quasi:  '                                        ]   ,...
    ['      base      = ' num2str(VmcOptions.quasi.base)        ]   ,...
    ['      dimension = ' num2str(VmcOptions.quasi.dimension)   ]   ,...
    ['      skip      = ' num2str(VmcOptions.quasi.skip)        ]   ,...
    ['  :stop quasi:    '                                       ]   ...
};
offset = offset+5;

% delete empty elements
VmcInput((offset+1):end) = [];

% write input file
fid = fopen(filename,'wt');
for i = 1 : length(VmcInput)
  fprintf(fid,'%s\n',VmcInput{i});
end
fclose(fid);

end