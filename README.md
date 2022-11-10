
# Iso-UVField
## Learning an Isometric Surface Parameterization for Texture Unwrapping (ECCV 2022)

[Project Page](https://sagniklp.github.io/isouvf/)

**This code is still in the cleaning phase. I expect to release a better version after Nov. 2nd. In the meantime, if you need urgent results or models, please contact me: sadas[at]cs[dot]stonybrook[dot]edu. TY!**

#### Environment Setup:

install the conda environment using env-isouvfield.yml


#### Experiments:
- you should use the configuration files in code/confs/ 
- edit the conf file to use any data_dirs (line 52 of the conf file) of your choice from '/input':
    e.g. paper3

- Run the experiments within the /code directory

#### Assets:
- The data can be downloaded from here: [GDrive Link](https://drive.google.com/drive/folders/12SwQ7vhz-VlgryZbT3o0ZYe9ECs6SOb4?usp=share_link)
- After downloading keep the data in '/input' directory.

#### Training:
* Step 1: exps/0000_00_00_00_00_00 folder contain pre-trained weights for the shape MLP, use it for faster training. Alternatively, you can use basic sphere initialization of IDR. To do that use 'pretrained' as the value of --timestamp flag 
	- Inital run with weighted backward no isometry:
		- command: ```python training/exp_runner_uvrend_colmap.py --nepoch 2000 --conf confs/doc_fixed_cameras_colmap_vanilla_idr_uvrend_agg.conf --gpu 0 --timestamp 0000_00_00_00_00_00 --is_continue --trainmode step1```

* Step 2: Run the following experiment after the basic experiment is done. Start from the 2000th epoch. You can use the timestamp of the basic experiment. Also remember to edit the config file to set the isometry weights and set the isometry flag to True. 
    - With isometry and weighted backward:
    	- command: ```python training/exp_runner_uvrend_colmap.py --nepoch 7000 --conf confs/doc_fixed_cameras_colmap_vanilla_idr_uvrend_agg.conf --gpu 0 --is_continue --timestamp <somefoldername> --trainmode step2```
* Step 3: Start from the 7000th epoch. You can use the timestamp of the previous experiment. Also remember to edit the config file to set the bwd loss mode and set the bwd weight flag to False. 
    - With isometry and backward L2:
    	- command: ```python training/exp_runner_uvrend_colmap.py --nepoch 8000 --conf confs/doc_fixed_cameras_colmap_vanilla_idr_uvrend_agg.conf --gpu 0 --is_continue --timestamp <somefoldername> --trainmode step3```
* All the steps sequentially: You don't have to worry about editing the ```config``` file in between.
    - command: ```python training/exp_runner_uvrend_colmap.py --nepoch 2000 --conf confs/doc_fixed_cameras_colmap_vanilla_idr_uvrend_agg.conf --gpu 0 --is_continue --timestamp 0000_00_00_00_00_00 --trainmode allsteps```

#### Evaluation:

Run the following command:
```python evaluation/unwarp_colmap.py --conf ./confs/doc_fixed_cameras_colmap_vanilla_idr_uvrend_agg.conf --eval_rendering --timestamp <somefoldername> --gpu 0 --blur --checkpoint 8000```

#### Release Notes:
- [X] Minimal training example (20th July, 2022) + error fixes (10th Nov, 2022)
- [ ] Code cleanup 
- [ ] Assets release
- [ ] Full code release
	- [ ] UV prior training code
	- [ ] Synthetic data training code
- [ ] Fix ReadMe
- [ ] Shell scripts for running experiments

#### Citation:
If you use the code, please consider citing our work-
```
@inproceedings{DasISOUVfield,
  author    = {Sagnik Das, Ke Ma, Zhixin Shu and Dimitris Samaras},
  title     = {Learning an Isometric Surface Parameterization for Texture Unwrapping},
  booktitle = {European Conference of Computer Vision 2022, {ECCV} 2022, Tel Aviv, Israel, October 23-27, 2022},
  publisher = {Springer},
  year      = {2022},
}
```
#### Acknowledgements:
This repository is heavily based on [IDR codebase](https://github.com/lioryariv/idr). Kudos to the contributors of IDR!
