import sys
sys.path.append('../code')
import argparse
import GPUtil
from pyhocon import ConfigFactory

from training.idruvrend_train import IDRTrainRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--trainmode', default='step1', type=str, help='Training mode')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--train_cameras', default=False, action="store_true", help='If set, optimizing also camera location.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
        
    parsed_conf=ConfigFactory.parse_file(opt.conf)
    if opt.trainmode=='step1':
        trainrunner1 = IDRTrainRunner(conf=opt.conf,
                                    parsed_conf=parsed_conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name='exps',
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    train_cameras=opt.train_cameras,
                                    sdf_init=True
                                    )

        trainrunner1.run()
        print ("\n\n..Round 1 training completed.. \n\n")
        
    elif opt.trainmode=='step2':
        # set the configs
        parsed_conf.put('loss.isom_weight',[0.1,0.1,0.1])
        parsed_conf.put('model.isometry',True)
        trainrunner2 = IDRTrainRunner(conf=opt.conf,
                                    parsed_conf=parsed_conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name='exps',
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    train_cameras=opt.train_cameras,
                                    sdf_init=False
                                    )

        trainrunner2.run()
        print ("\n\n..Round 2 training completed.. \n\n")
        
    elif opt.trainmode=='step3':
        # set the configs
        parsed_conf.put('loss.isom_weight',[0.1,0.1,0.1])
        parsed_conf.put('model.isometry',True)
        parsed_conf.put('loss.bwd_mode','l2')
        parsed_conf.put('dataset.imp_map',False)
         
        trainrunner3 = IDRTrainRunner(conf=opt.conf,
                                    parsed_conf=parsed_conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name='exps',
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    train_cameras=opt.train_cameras,
                                    sdf_init=False
                                    )

        trainrunner3.run()
        print ("\n\n..Round 3 training completed.. \n\n")
        
    elif opt.trainmode=='allsteps':
        trainrunner1 = IDRTrainRunner(conf=opt.conf,
                            parsed_conf=parsed_conf,
                            batch_size=opt.batch_size,
                            nepochs=opt.nepoch,
                            expname=opt.expname,
                            gpu_index=gpu,
                            exps_folder_name='exps',
                            is_continue=opt.is_continue,
                            timestamp=opt.timestamp,
                            checkpoint=opt.checkpoint,
                            train_cameras=opt.train_cameras,
                            sdf_init=True
                            )

        trainrunner1.run()
        print ("\n\n..Round 1 training completed.. \n\n")
        
        # set the configs
        new_timestamp=trainrunner1.timestamp
        parsed_conf.put('loss.isom_weight',[0.1,0.1,0.1])
        parsed_conf.put('model.isometry',True)
        trainrunner2 = IDRTrainRunner(conf=opt.conf,
                                    parsed_conf=parsed_conf,
                                    batch_size=opt.batch_size,
                                    nepochs=7000,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name='exps',
                                    is_continue=opt.is_continue,
                                    timestamp=new_timestamp,
                                    checkpoint=opt.checkpoint,
                                    train_cameras=opt.train_cameras,
                                    sdf_init=False
                                    )

        trainrunner2.run()
        print ("\n\n..Round 2 training completed.. \n\n")
        
        # set the configs
        new_timestamp=trainrunner2.timestamp
        parsed_conf.put('loss.isom_weight',[0.1,0.1,0.1])
        parsed_conf.put('model.isometry',True)
        parsed_conf.put('loss.bwd_mode','l2')
        parsed_conf.put('dataset.imp_map',False)
         
        trainrunner3 = IDRTrainRunner(conf=opt.conf,
                                    parsed_conf=parsed_conf,
                                    batch_size=opt.batch_size,
                                    nepochs=8000,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name='exps',
                                    is_continue=opt.is_continue,
                                    timestamp=new_timestamp,
                                    checkpoint=opt.checkpoint,
                                    train_cameras=opt.train_cameras,
                                    sdf_init=False
                                    )

        trainrunner3.run()
        print ("\n\n..Round 3 training completed.. \n\n")
        
    else:
        print ("Specify the training mode: [step1/step2/step3/allsteps]")