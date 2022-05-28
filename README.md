# MetaPlanner

---

MetaPlanner is an open source automated treatment planning method that performs meta-optimization of treatment planning hyperparameters. It is meant for educational/research purposes and supports planning in MatRad using one of their experimental branches. 

If you find this project useful, please cite our [work](https://iopscience.iop.org/article/10.1088/1361-6560/ac5672):
```
@misc{huang2021metaoptimization,
      title={Meta-optimization for Fully Automated Radiation Therapy Treatment Planning}, 
      author={Charles Huang and Yusuke Nomura and Yong Yang and Lei Xing},
      year={2021},
      eprint={2110.10733},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph}
}
```
---

All source code for MetaPlanner and MatRad are provided under the terms of the GNU GENERAL PUBLIC LICENSE Version 3 (GPL v3). The precompiled mex files of the Ipopt optimizer in object code version are licensed under the Eclipse Public License Version 1.0 (EPL v1.0).

---

Basic Examples:

Begin by storing your CT dicoms and RtSt dicoms into the following structure:

	.
	├── /head_and_neck_src
	│   ├── /head_and_neck_data/                    
	│   ├──   ├── /CT/                    
	│   ├──   ├── /RTst/
	├── /prostate_src
	│   ├── /prostate_data/                    
	│   ├──   ├── /CT/                    
	│   ├──   ├── /RTst/
	└── ...

To run automated planning, simply modify and run [this](https://github.com/chh105/MetaPlanner/blob/main/head_and_neck_src/run_meta_optimization_framework.py) script for head and neck data, or [this](https://github.com/chh105/MetaPlanner/blob/main/prostate_src/run_meta_optimization_framework.py) script for prostate data.