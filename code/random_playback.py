# play WAV file on loop (current set up: plays from tsippor through speaker in box 2)
#
# arguments: 
#   - path to sound file (str)
#   - number of times to play sound (int)
#   - total play time in minutes (int)
#
# written for songs < 30 seconds in length
#
# example use: python ~/scripts/random_playback.py /path/to/sound.wav 100 60

import time
import random
import pygame
import wave
import argparse

def play_sound(file_path):
    pygame.mixer.init()  # initialize pygame mixer
    pygame.mixer.music.load(file_path)  # load the sound file
    pygame.mixer.music.play()  # play the sound
    while pygame.mixer.music.get_busy():  # wait for the sound to finish playing
        time.sleep(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Play WAV file on loop.')
    parser.add_argument('file_path', type=str, nargs='?', default='/home/finch/stim_files/z_c4y1_23/z_c4y1_23_2024-02-10_song62.wav',
                        help='Path to the WAV file.')
    parser.add_argument('n_plays', type=int, nargs='?', default=100, help='Number of times to play sound (default: 100).')
    parser.add_argument('total_time', type=int, nargs='?', default=60, help='Total play time (default: 60 minutes).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    file_path = args.file_path  # WAV file path
    n_plays = args.n_plays  # number of plays
    total_time = args.total_time  # total play time
    
    # get wav file length (in seconds)
    wav_file = wave.open(file_path,'r')
    num_frames = wav_file.getnframes()
    frame_rate = wav_file.getframerate()
    wav_len = num_frames / float(frame_rate)
    wav_file.close()
    
    # initialize remaining play time and number of plays
    remaining_play_time = total_time * 60
    remaining_plays = n_plays
    
    # play at randomized intervals
    while remaining_play_time > 0 and remaining_plays > 0:
        play_sound(file_path)
        
        # calculate average interval based on remaining time and plays
        average_interval = remaining_play_time / remaining_plays - wav_len
        
        # randomize interval
        interval = random.uniform(average_interval * 0.5, average_interval * 1.5)
        
        # adjust remaining play time and plays
        remaining_play_time -= (interval + wav_len)
        remaining_plays -= 1
        
        time.sleep(interval)
