
import numpy as np
import os
from pyradigm import MLDataset

import neuropredict
import rhst
import freesurfer

feat_generator = np.random.randn

out_dir = os.path.abspath('../tests')
meta_file = os.path.join(out_dir,'meta.csv')

meta = list()

def make_random_MLdataset(max_num_classes = 20,
                          max_class_size = 50,
                          max_dim = 100,
                          stratified = True):
    "Generates a random MLDataset for use in testing."

    num_classes = np.random.randint(2, max_num_classes, 1)
    if not stratified:
        class_sizes = np.random.random_integers(10, max_class_size, size=[num_classes, 1])
    else:
        class_sizes = np.repeat(np.random.randint(10, max_class_size), num_classes)

    num_features = np.random.randint(1, max_dim, 1)[0]

    class_ids = list()
    labels = list()
    for cl in range(num_classes):
        class_ids.append('class-{}'.format(cl))
        labels.append(int(cl))

    ds = MLDataset()
    for cc, class_ in enumerate(class_ids):
        subids = [ 'sub{:3}-class{:3}'.format(ix,cc) for ix in range(class_sizes[cc]) ]
        for sid in subids:
            ds.add_sample(sid, feat_generator(num_features), int(cc), class_)


    return ds


random_dataset = make_random_MLdataset( max_num_classes = 3)
class_set, label_set, class_sizes = random_dataset.summarize_classes()

out_path = os.path.join(out_dir, 'random_dataset.pkl')
random_dataset.save(out_path)

out_list = os.path.join(out_dir, 'list_datasets.txt')
with open(out_list, 'w') as lf:
    lf.writelines('\n'.join([out_path, ]))

res_path = rhst.run(out_list, out_dir, num_repetitions=20)

dataset_paths, train_perc, num_repetitions, num_classes, \
           pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
           best_min_leaf_size, best_num_predictors, feature_importances_rf, \
           num_times_misclfd, num_times_tested, \
           confusion_matrix, class_set, accuracy_balanced, auc_weighted = rhst.load_results(res_path)

print ''


# TODO accuracy and AUC on random binary datasets must be chance!
# assert np.median(accuracy_balanced) == np.median(rhst.chance_accuracy(class_sizes))