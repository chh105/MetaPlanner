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

---
Environment

Please setup the environment using either (1) the yml file or (2) the copy [here](https://drive.google.com/file/d/12JXcyQg90emWFlAe_H8xPwKEhCy_3Fek/view?usp=drive_link). If you choose to use (2), please follow the conda unpack instructions below:
```
# Unpack environment into directory `my_env`
$ mkdir -p my_env
$ tar -xzf my_env.tar.gz -C my_env

# Use Python without activating or fixing the prefixes. Most Python
# libraries will work fine, but things that require prefix cleanups
# will fail.
$ ./my_env/bin/python

# Activate the environment. This adds `my_env/bin` to your path
$ source my_env/bin/activate

# Run Python from in the environment
(my_env) $ python

# Cleanup prefixes from in the active environment.
# Note that this command can also be run without activating the environment
# as long as some version of Python is already installed on the machine.
(my_env) $ conda-unpack
```
