### Spike sort
#
# Script within the chronic ephys processing pipeline
# - 1-preprocess_acoustics
# - 2-curate_acoustics
# - **3-sort_spikes**
# - 4-curate_spikes
#
# Use the environment **spikeproc** to run this notebook


## Import packages
import numpy as np
import os
import pickle
os.environ["NPY_MATLAB_PATH"] = '/mnt/cube/chronic_ephys/code/npy-matlab'
os.environ["KILOSORT2_PATH"] = '/mnt/cube/chronic_ephys/code/Kilosort2'
os.environ["KILOSORT3_PATH"] = '/mnt/cube/chronic_ephys/code/Kilosort'
import spikeinterface.full as si
import sys
import traceback
import torch
sys.path.append('/mnt/cube/lo/envs/ceciestunepipe/')
from ceciestunepipe.file import bcistructure as et
from ceciestunepipe.mods import probe_maps as pm


## Set parameters
si.get_default_sorter_params('kilosort4')

# non default spike sorting parameters
sort_params_dict_ks3 = {'minFR':0.001, 'minfr_goodchannels':0.001} # kilosort 3
sort_params_dict_ks4_npx = {'batch_size':30000, 'nblocks':5, 'Th_universal':8, 'Th_learned':7, 'dmin':15, 'dminx':32} # kilosort 4, neuropixels (set dmin and dminx to true pitch)
sort_params_dict_ks4_nnx64 = {'nblocks':0, 'nearest_templates':64,
                              'Th_universal':8, 'Th_learned':7} # kilosort 4, neuronexus 64 chan

# waveform extraction parameters
wave_params_dict = {'ms_before':1, 'ms_after':2, 'max_spikes_per_unit':500,
                    'sparse':True, 'num_spikes_for_sparsity':100, 'method':'radius',
                    'radius_um':40, 'n_components':5, 'mode':'by_channel_local'}

# print stuff
verbose = True

# errors break sorting
raise_error = False

# restrict sorting to a specific GPU
restrict_to_gpu = 0 # None

# use specific GPU if specified
if restrict_to_gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(restrict_to_gpu)

# parallel processing params
job_kwargs = dict(n_jobs=28,chunk_duration="1s",progress_bar=False)
si.set_global_job_kwargs(**job_kwargs)

# force processing of previous failed sorts
skip_failed = False

# session info
bird_rec_dict = {
    'z_g10r16_24':[
        {'sess_par_list':['2024-09-25'], # sessions (will process all epochs within)
         'probe':{'probe_type':'neuropixels-2.0'}, # probe specs
         'sort':'sort_0', # label for this sort instance
         'sorter':'kilosort4', # sort method
         'sort_params':sort_params_dict_ks4_npx, # non-default sort params
         'wave_params':wave_params_dict, # waveform extraction params
         'ephys_software':'sglx' # sglx or oe
        },
    ],
}



## Run sorts

# store sort summaries
sort_summary_all = []

# loop through all birds / recordings
for this_bird in bird_rec_dict.keys():
    # get session configurations
    sess_all = bird_rec_dict[this_bird]
    
    # loop through session configurations
    for this_sess_config in sess_all:
        
        # loop through sessions
        for this_sess in this_sess_config['sess_par_list']:
            log_dir = os.path.join('/mnt/cube/chronic_ephys/log', this_bird, this_sess)
            
            # build session parameter dictionary
            sess_par = {'bird':this_bird,
                        'sess':this_sess,
                        'ephys_software':this_sess_config['ephys_software'],
                        'sorter':this_sess_config['sorter'],
                        'sort':this_sess_config['sort']}
            # get epochs
            sess_epochs = et.list_ephys_epochs(sess_par)
            sess_epochs = ['0701_g0', '0925_g0']
            concat_epochs = '-'.join(sess_epochs) # process all epochs together -- edit to concatenate only some if desired
            
            # set output directory
            epoch_struct = et.sgl_struct(sess_par,concat_epochs,ephys_software=sess_par['ephys_software'])
            sort_path = epoch_struct['folders']['derived'] + '/{}/{}/'.format(sess_par['sorter'],sess_par['sort'])
            sorting_analyzer_path = sort_path + 'sorting_analyzer/'

            # get spike sort log
            try:
                with open(os.path.join(log_dir,concat_epochs+'_spikesort_'+this_sess_config['sort']+'.log'), 'r') as f:
                    log_message=f.readline() # read the first line of the log file
                if log_message[:-1] == sess_par['bird']+' '+sess_par['sess']+' '+this_epoch+' sort complete without error':
                    print(sess_par['bird'],sess_par['sess'],this_epoch,'already exists -- skipping sort')
                    run_proc = False
                elif log_message[:-1] == sess_par['bird']+' '+sess_par['sess']+' '+this_epoch+' sort failed':
                    if skip_failed:
                        print(sess_par['bird'],sess_par['sess'],this_epoch,'previously failed -- skipping sort')
                        run_proc = False
                    else:
                        run_proc = True
                else: # uninterpretable log file
                    run_proc = True
            except: # no existing log file
                run_proc = True

            # run sort
            if run_proc:
                try:
                    print('___________',this_bird,this_sess,concat_epochs,'___________')
                    # prepare recording for sorting
                    print('prep..')
                    if sess_par['ephys_software'] == 'sglx':
                        # load recordings
                        rec_path = '/'.join(epoch_struct['folders']['sglx'].split('/')[:-1])
#                         epoch_list = [e for e in os.listdir(rec_path) if os.path.isdir(os.path.join(rec_path,e))]
                        recording_list = []
                        for this_epoch in sess_epochs:
                            this_rec = si.read_spikeglx(folder_path=os.path.join(rec_path,this_epoch), stream_name='imec0.ap')
                            probe_df = this_rec.get_probe().to_dataframe()
                            # ibl destriping
                            this_rec = si.highpass_filter(recording=this_rec)
                            this_rec = si.phase_shift(recording=this_rec)
                            bad_good_channel_ids = si.detect_bad_channels(recording=this_rec)
                            if len(bad_good_channel_ids[0]) > 0:
                                this_rec = si.interpolate_bad_channels(recording=this_rec,bad_channel_ids=bad_good_channel_ids[0])
                            if this_sess_config['probe']['probe_type'] == 'neuropixels-2.0':
                                # highpass by shank
                                split_rec = this_rec.split_by(property='group',outputs='list') # split recording by shank
                                split_rec = [si.highpass_spatial_filter(recording=r,n_channel_pad=min(r.get_num_channels(),60)) for r in split_rec]
                                this_rec = si.aggregate_channels(split_rec) # recombine shanks
                                # stack shanks
                                p,_ = pm.stack_shanks(probe_df) # make new Probe object with shanks stacked
                                this_rec = this_rec.set_probe(p,group_mode='by_probe') # assign new Probe object to probe
                            else:
                                this_rec = si.highpass_spatial_filter(recording=this_rec)
                            recording_list.append(this_rec)
                        
                        # concatenate recordings
                        this_rec_p = si.concatenate_recordings(recording_list)
                        # save probe map prior to re-ordering for sorting
                        probe_df.to_pickle(os.path.join(epoch_struct['folders']['derived'],'probe_map_df.pickle'))
                    else:
                        raise TypeError('Only implemented for sglx')
                    # set sort params
                    this_rec_p = si.concatenate_recordings([this_rec_p])
                    sort_params = si.get_default_sorter_params(this_sess_config['sorter'])
                    for this_param in this_sess_config['sort_params'].keys():
                        sort_params[this_param] = this_sess_config['sort_params'][this_param]
                    # run sort
                    print('sort..')
                    torch.cuda.empty_cache()
                    this_sort = si.run_sorter(sorter_name=this_sess_config['sorter'],recording=this_rec_p,output_folder=sort_path,
                                         remove_existing_folder=True,delete_output_folder=False,delete_container_files=False,
                                         verbose=verbose,raise_error=raise_error,**sort_params)
                    torch.cuda.empty_cache()
                    # bandpass recording before running analyzer
                    this_rec_pf = si.bandpass_filter(recording=this_rec_p)
                    # run sorting analyzer
                    print('sorting analyzer..')
                    analyzer = si.create_sorting_analyzer(sorting=this_sort,recording=this_rec_pf,format="binary_folder",
                                                          sparse=True,return_scaled=True,folder=sorting_analyzer_path)
                    ext_compute_all = analyzer.get_computable_extensions()
                    for this_ext in ext_compute_all:
                        print(this_ext + '..')
                        analyzer.compute(this_ext)
                    
                    # mark complete
                    print('COMPLETE!!')

                    # log complete sort
                    if not os.path.exists(log_dir): os.makedirs(log_dir)
                    with open(os.path.join(log_dir,concat_epochs+'_spikesort_'+this_sess_config['sort']+'.log'), 'w') as f:
                        f.write(sess_par['bird']+' '+sess_par['sess']+' sort complete without error\n\n')
                        f.write('Sort method: '+this_sess_config['sorter']+'\n\n')
                        f.write('Sort params: '+str(sort_params)+'\n\n')
                        f.write('Computable extensions: '+str(ext_compute_all)+'\n\n')
                    sort_summary = [this_bird,this_sess,sess_par['ephys_software'],concat_epochs,'COMPLETE']

                except Exception as e:
                    # mark exception
                    print("An exception occurred:", e)

                    # log failed sort
                    if not os.path.exists(log_dir): os.makedirs(log_dir)
                    with open(os.path.join(log_dir,concat_epochs+'_spikesort_'+this_sess_config['sort']+'.log'), 'w') as f:
                        f.write(sess_par['bird']+' '+sess_par['sess']+' sort failed\n')
                        f.write(traceback.format_exc())
                    sort_summary = [this_bird,this_sess,sess_par['ephys_software'],concat_epochs,'FAIL']
            else:
                sort_summary = [this_bird,this_sess,sess_par['ephys_software'],concat_epochs,'EXISTS']

            # report and store sort summary
            print(sort_summary)
            sort_summary_all.append(sort_summary)