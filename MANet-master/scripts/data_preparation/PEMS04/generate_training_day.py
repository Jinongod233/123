import os
import sys
import argparse

import numpy as np

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))


def generate_data(args: argparse.Namespace):


    target_channel = args.target_channel
    output_dir = args.output_dir
    data_file_path = args.data_file_path
    save =args.save
    num_of_hours = args.num_of_hours
    num_of_days = args.num_of_days
    num_of_weeks = args.num_of_weeks
    steps_per_day = args.steps_per_day
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    # read data
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    # add external feature
    l, n, f = data.shape
    index_list = []
    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day /
               steps_per_day for i in range(data.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        index_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        index_list.append(dow_tiled)

    processed_data = np.concatenate(index_list, axis=-1)


    def search_data(sequence_length, num_of_depend, label_start_idx,
                    num_for_predict, units, points_per_hour):

        if points_per_hour < 0:
            raise ValueError("points_per_hour should be greater than 0!")

        if label_start_idx + num_for_predict > sequence_length:
            return None

        x_idx = []
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - points_per_hour * units * i
            end_idx = start_idx + num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None

        if len(x_idx) != num_of_depend:
            return None

        return x_idx[::-1]



    def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                           label_start_idx, num_for_predict, points_per_hour=12):
        week_sample, day_sample, hour_sample = None, None, None

        if label_start_idx + num_for_predict > data_sequence.shape[0]:
            return week_sample, day_sample, hour_sample, None

        if num_of_weeks > 0:
            week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                       label_start_idx, num_for_predict,
                                       7 * 24, points_per_hour)
            if not week_indices:
                return None, None, None, None

            week_sample = np.concatenate([data_sequence[i: j]
                                          for i, j in week_indices], axis=0)

        if num_of_days > 0:
            day_indices = search_data(data_sequence.shape[0], num_of_days,
                                      label_start_idx, num_for_predict,
                                      24, points_per_hour)
            if not day_indices:
                return None, None, None, None

            day_sample = np.concatenate([data_sequence[i: j]
                                         for i, j in day_indices], axis=0)

        if num_of_hours > 0:
            hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                       label_start_idx, num_for_predict,
                                       1, points_per_hour)  # 选取一段区间
            if not hour_indices:
                return None, None, None, None

            hour_sample = np.concatenate([data_sequence[i: j]
                                          for i, j in hour_indices], axis=0)

        target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

        return week_sample, day_sample, hour_sample, target

    # def MinMaxnormalization(train, val, test):
    #
    #     assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[
    #                                                                  1:]  # ensure the num of nodes is the same
    #
    #     _max = train.max(axis=(0, 1, 3), keepdims=True)
    #     _min = train.min(axis=(0, 1, 3), keepdims=True)
    #
    #     print('_max.shape:', _max.shape)
    #     print('_min.shape:', _min.shape)
    #
    #     def normalize(x):
    #         x = 1. * (x - _min) / (_max - _min)
    #         x = 2. * x - 1.
    #         return x
    #
    #     train_norm = normalize(train)
    #     val_norm = normalize(val)
    #     test_norm = normalize(test)
    #
    #     return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


    all_samples = []
    for idx in range(processed_data.shape[0]):
        sample = get_sample_indices(processed_data, num_of_weeks, num_of_days,
                                    num_of_hours, idx, 12,
                                    12)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0)  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0)  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0) # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0)  #[:, :, 0, :]   (1,N,F,T)
        sample.append(target)

        # time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        # sample.append(time_sample)

        all_samples.append(
            sample)


    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre)],(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-1],axis=1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
    val_x = np.concatenate(validation_set[:-1], axis=1)
    test_x = np.concatenate(testing_set[:-1], axis=1)

    train_target = training_set[-1]  # (B,N,T)
    val_target = validation_set[-1]
    test_target = testing_set[-1]


    all_data = {
        'train': {
            'x': train_x,
            'target': train_target,
            # 'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x,
            'target': val_target,
            # 'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x,
            'target': test_target,
            # 'timestamp': test_timestamp,
        },
        # 'stats': {
        #     '_max': stats['_max'],
        #     '_min': stats['_min'],
        # }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)

    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)

    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)

    print()


    file = os.path.basename(output_dir).split('.')[0]
    dir_path = os.path.dirname(output_dir)

    if save:

        filename = os.path.join(dir_path,
                                file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) +'_index')
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            )
    return all_data



if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence

    TARGET_CHANNEL = [0]                   # target channel(s)
    STEPS_PER_DAY = 288

    SAVE = False
    #741
    NUM_OF_HOURS = 7
    NUM_OF_DAYS = 4
    NUM_OF_WEEKS = 1

    DATASET_NAME = "PEMS04"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    OUTPUT_DIR = "datasets/" + DATASET_NAME + "/PEMS04.npz"
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--save", type=bool,
                        default=True, help="Save.")
    parser.add_argument("--num_of_hours", type=bool,
                        default=NUM_OF_HOURS, help="Num_of_hours.")
    parser.add_argument("--num_of_days", type=bool,
                        default=NUM_OF_DAYS, help="Num_of_days.")
    parser.add_argument("--num_of_weeks", type=bool,
                        default=NUM_OF_WEEKS, help="Num_of_weeks.")
    args_metr = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args_metr).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))


    generate_data(args_metr)
