import os
import cv2
import pickle
import pandas as pd
import numpy as np
import json


# Generate frameidx for shot/event change
def generate_frameidx_from_raw(min_change_duration=0.3, split='valnew'):
    assert split in ['train', 'val', 'valnew', 'test']

    with open('../export/k400_{}_raw_annotation.pkl'.format(split), 'rb') as f:
        dict_raw = pickle.load(f, encoding='lartin1')

    mr345 = {}
    for filename in dict_raw.keys():
        ann_of_this_file = dict_raw[filename]['substages_timestamps']
        if not (len(ann_of_this_file) >= 3):
            # print(f'{filename} less than 3 annotations.')
            continue

        try:
            fps = dict_raw[filename]['fps']
            num_frames = int(dict_raw[filename]['num_frames'])
            video_duration = dict_raw[filename]['video_duration']
            avg_f1 = dict_raw[filename]['f1_consis_avg']
        except:
            # print(f'{filename} exception!')
            continue

        # this is avg f1 from halo but computed using the annotation after post-processing like merge two close changes
        mr345[filename] = {}
        mr345[filename]['num_frames'] = int(dict_raw[filename]['num_frames'])
        mr345[filename]['path_video'] = dict_raw[filename]['path_video']
        mr345[filename]['fps'] = dict_raw[filename]['fps']
        mr345[filename]['video_duration'] = dict_raw[filename]['video_duration']
        mr345[filename]['path_frame'] = dict_raw[filename]['path_video'].split('.mp4')[0]
        mr345[filename]['f1_consis'] = []
        mr345[filename]['f1_consis_avg'] = avg_f1

        mr345[filename]['substages_myframeidx'] = []
        mr345[filename]['substages_timestamps'] = []
        for ann_idx in range(len(ann_of_this_file)):
            # remove changes at the beginning and end of the video;
            ann = ann_of_this_file[ann_idx]
            tmp_ann = []
            change_shot_range_start = []
            change_shot_range_end = []
            change_event = []
            change_shot_timestamp = []
            for p in ann:
                st = p['start_time']
                et = p['end_time']
                l = p['label'].split(' ')[0]
                if (st + et) / 2 < min_change_duration or (st + et) / 2 > (
                        video_duration - min_change_duration): continue
                tmp_ann.append(p)
                if l == 'EventChange':
                    change_event.append((st + et) / 2)
                elif l == 'ShotChangeGradualRange:':
                    change_shot_range_start.append(st)
                    change_shot_range_end.append(et)
                else:
                    change_shot_timestamp.append((st + et) / 2)

            # consolidate duplicated/very close timestamps
            # if two shot range overlap, merge
            i = 0
            while i < len(change_shot_range_start) - 1:
                while change_shot_range_end[i] >= change_shot_range_start[i + 1]:
                    change_shot_range_start.remove(change_shot_range_start[i + 1])
                    if change_shot_range_end[i] <= change_shot_range_end[i + 1]:
                        change_shot_range_end.remove(change_shot_range_end[i])
                    else:
                        change_shot_range_end.remove(change_shot_range_end[i + 1])
                    if i == len(change_shot_range_start) - 1:
                        break
                i += 1

                # if change_event or change_shot_timestamp falls into range of shot range, remove this change_event
            for cg in change_event:
                for i in range(len(change_shot_range_start)):
                    if cg <= (change_shot_range_end[i] + min_change_duration) and cg >= (
                            change_shot_range_start[i] - min_change_duration):
                        change_event.remove(cg)
                        break
            for cg in change_shot_timestamp:
                for i in range(len(change_shot_range_start)):
                    if cg <= (change_shot_range_end[i] + min_change_duration) and cg >= (
                            change_shot_range_start[i] - min_change_duration):
                        change_shot_timestamp.remove(cg)
                        break

            # if two timestamp changes are too close, remove the second one between two shot changes, two event changes; shot vs. event, remove event
            change_event.sort()
            change_shot_timestamp.sort()
            tmp_change_shot_timestamp = change_shot_timestamp
            tmp_change_event = change_event
            # """
            i = 0
            while i <= (len(change_event) - 2):
                if (change_event[i + 1] - change_event[i]) <= 2 * min_change_duration:
                    tmp_change_event.remove(change_event[i + 1])
                else:
                    i += 1
            i = 0
            while i <= (len(change_shot_timestamp) - 2):
                if (change_shot_timestamp[i + 1] - change_shot_timestamp[i]) <= 2 * min_change_duration:
                    tmp_change_shot_timestamp.remove(change_shot_timestamp[i + 1])
                else:
                    i += 1
            for i in range(len(tmp_change_shot_timestamp) - 1):
                j = 0
                while j <= (len(tmp_change_event) - 1):
                    if abs(tmp_change_shot_timestamp[i] - tmp_change_event[j]) <= 2 * min_change_duration:
                        tmp_change_event.remove(tmp_change_event[j])
                    else:
                        j += 1
            # """
            change_shot_timestamp = tmp_change_shot_timestamp
            change_event = tmp_change_event
            change_shot_range = []
            for i in range(len(change_shot_range_start)):
                change_shot_range += [(change_shot_range_start[i] + change_shot_range_end[i]) / 2]

            change_all = change_event + change_shot_timestamp + change_shot_range
            change_all.sort()
            time_change_all = change_all

            change_all = np.floor(np.array(change_all) * fps)
            tmp_change_all = []
            for cg in change_all:
                tmp_change_all += [min(num_frames - 1, cg)]

            # if len(tmp_change_all) != 0: #even after processing, the list is empty/there is no GT bdy, shall still keep []
            mr345[filename]['substages_myframeidx'] += [tmp_change_all]
            mr345[filename]['substages_timestamps'] += [time_change_all]
            mr345[filename]['f1_consis'] += [dict_raw[filename]['f1_consis'][ann_idx]]

    with open(f'../export/k400_mr345_{split}_min_change_duration{min_change_duration}.pkl', 'wb') as f:
        pickle.dump(mr345, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(len(mr345))


if __name__ == '__main__':
    generate_frameidx_from_raw(split='train')
    # generate_frameidx_from_raw(split='test')
    generate_frameidx_from_raw(split='val')
    # generate_frameidx_from_raw(split='valnew')
