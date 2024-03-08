# Chronic ephys pipeline documentation

Pipeline built in the Gentner Lab that takes in chronic single-unit electrophysiology, audio data, and other forms of data streams, syncs streams together, and handles automated and hand curated spike sorting and song analysis. The following notebooks are provided:

## 1-preprocess_acoustics

This notebook handles the initial preprocessing of neural data from SpikeGLX and OpenEphys as well as other simultaneous data streams (audio or pressure).

Environment to use: **songprep**. This notebook and *2-curate_acoustics* rely on Zeke Arneodo's pipeline <u>ceciestunepipe</u>. [Ceciestunepipe](https://github.com/laurenostrowski/ceciestunepipe) handles loading and syncing neural and audio data, but requires certain early versions of published packages that have since been updated. The environment **songprep** is compatible with ceciestunepipe, and is therefore used for preprocessing steps, but should not be used for forward analyses.

#### Errors you might encounter
- **Number of edges in the syn ttl events of pattern and target don't match:** sy.sync_all will throw an error if streams are different lengths. For example, if the neuropixel comes unplugged during a recording session, NIDQ and WAV streams will be of the same length, different than the length of LFP and AP streams (see bird: z_c5o30_23, sess: 2023-08-12 for an example). This is because NIDQ contains a microphone channel that will continue recording even if the neuropixel data is disrupted.
- **Events array for lf_0 had skipped heartbeats:** sy.sync_all will throw an error if it detects that heartbeats were skipped in any of the data streams. We have configured our system to send a 0.5 Hz square wave pulse to align the machine clocks for all data streams. If it detects periods between square wave pulses (i.e., heartbeats) are of different lengths, it might mean that the signal was momentarily lost.

## 2-curate_acoustics

This notebook provides an interface for you to manually curate automatically detected bouts of song. You can manually inspect putative bouts and trim the bout start and stop indices (precise alignment to the bout will improve the performance of the syllable curation algorithm later on).

Environment to use: **songprep**.

## 3-sort_spikes

This notebook runs an automatic spike sorting algorithm on the neural data, defaulting to the spike sorting algorithm [Kilosort 3](https://kilosort.readthedocs.io/en/latest/). *Kilosort 4 was released on March 3rd, 2024 and will be added to the notebook as the default sorting algorithm when integrated into SpikeInterface*.

Environment to use: **spikeprep**.

## 4-curate_spikes

This notebook allows you to manually curate the outputs of an automatic spike sorting algorithm using the web-based viewer provided by [SpikeInterface](https://spikeinterface.readthedocs.io/en/latest/).

Environment to use: **spikeprep**.

#### Errors you might encounter
- **ValueError: Out of range float values are not JSON compliant:** you might get an error message resembling this when running *si.plot_quality_metrics* if one or more quality metrics could not be computed in automatic spike sorting. Skip the metrics that could not be computed (using the spike sort log as a guide) using the flag skip_metrics=['metrics', 'to', 'skip']

## 5-cluster_acoustics

This *optional* notebook...

## 6-convert_to_NWB

This notebook converts the outputs of processing pipeline to the [Neuroscience without Borders (NWB) Data Standard](https://www.nwb.org/nwb-neurophysiology/) formatting.
