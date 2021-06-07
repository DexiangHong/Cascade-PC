import pickle
import numpy as np
import argparse


def get_idx_from_score_by_threshold(threshold=0.5, seq_indices=None, seq_scores=None):
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices = []
    internals_indices = []
    for i in range(len(seq_scores)):
        if seq_scores[i] >= threshold:
            internals_indices.append(i)
        elif seq_scores[i] < threshold and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)
            internals_indices = []

        if i == len(seq_scores) - 1 and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)

    bdy_indices_in_video = []
    if len(bdy_indices) != 0:
        for internals in bdy_indices:
            center = int(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
    return bdy_indices_in_video


parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, default='../data/export/k400_mr345_val_min_change_duration0.3.pkl')
parser.add_argument('--pred', type=str, default='multif-pred_outputs/k400_checkpoint_release.pth.tar_val_5_kinetics_multiframes_scores.pkl')
parser.add_argument('--save', type=str, default='submission.pkl')
args = parser.parse_args()

with open(args.gt, 'rb') as f:
    gt_dict = pickle.load(f, encoding='lartin1')

with open(args.pred, 'rb') as f:  #
    my_pred = pickle.load(f, encoding='lartin1')

print(len(gt_dict))
save = dict()
for vid in my_pred:
    if vid in gt_dict:
        # detect boundaries, convert frame_idx to timestamps
        fps = gt_dict[vid]['fps']
        det_t = np.array(get_idx_from_score_by_threshold(threshold=0.5,
                                                         seq_indices=my_pred[vid]['frame_idx'],
                                                         seq_scores=my_pred[vid]['scores'])) / fps
        save[vid] = det_t.tolist()
print(len(save))
pickle.dump(save, open(args.save, 'wb'), protocol=4)
