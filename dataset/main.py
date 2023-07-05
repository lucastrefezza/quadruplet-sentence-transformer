import argparse
import os
import math
from dataset.constants import COCO_CAPTIONS_TRAIN, COCO_CAPTIONS_VAL, CLEANED_COCO_TRAIN, CLEANED_COCO_VAL, \
    CLEANED_COCO_TEST, COCO_DS_TYPE, SENTENCE_SUMMARIZATION_DS_TYPE, RAW_DATA, COCO_DS_NAME, CHUNK_DIM, N_EXAMPLES, \
    N_PART_EXAMPLES, SIMILARITY_THRESHOLD
from dataset.coco_dataset_creation import create_coco_dataset, CocoCaptionsOnly
from dataset.quadruplet_dataset import QuadrupletDataset, RANDOM


def main(args):
    if args.dataset_type == COCO_DS_TYPE:

        print("Creating torch coco caption datasets...")
        cap_train = CocoCaptionsOnly(
            root=os.path.join(args.raw_data_root, "images"),
            dataset_name=args.dataset_name + "_train",
            ann_file=args.raw_train_path,
            transform=None
        )
        cap_test = CocoCaptionsOnly(
            root=os.path.join(args.raw_data_root, "images"),
            dataset_name=args.dataset_name + "_test",
            ann_file=args.raw_val_path,
            transform=None
        )

        if args.verbose_check:
            print("Number of samples train: ", len(cap_train))
            print('Number of samples: ', len(cap_test))
            img, target = cap_test[3]  # load 4th sample

            print("Image Size: ", img.size)
            print(target)
            img, target = cap_train[4]  # load 5th sample

            print("Image Size: ", img.size)
            print(target)

            print(cap_train.ann_file)
            print(cap_train.dataset_name)

        print(f"Total chunk number test: {math.ceil(len(cap_test) / args.chunk_dim)}")
        print(f"Total chunk number train: {math.ceil(len(cap_train) / args.chunk_dim)}")
        print(f"Creating test dataset in {args.test_out_path}")
        create_coco_dataset(
            root=args.test_out_path,
            dataset=cap_test,
            start_chunk=args.start_chunk_test,
            last_chunk=args.last_chunk_test if args.last_chunk_test > 0 else None,
            chunk_dim=args.chunk_dim,
            n_pos_examples=args.n_pos_examples,
            n_part_pos_examples=args.n_part_pos_examples,
            sim_threshold=args.pos_sim_threshold,
            augment=True,
            log_tqdm=args.verbose_creation
        )
        print(f"Creating train dataset in {args.train_out_path}")
        create_coco_dataset(
            root=args.train_out_path,
            dataset=cap_train,
            start_chunk=args.start_chunk_train,
            last_chunk=args.last_chunk_train if args.last_chunk_train > 0 else None,
            chunk_dim=args.chunk_dim,
            n_pos_examples=args.n_pos_examples,
            n_part_pos_examples=args.n_part_pos_examples,
            sim_threshold=args.pos_sim_threshold,
            augment=True,
            log_tqdm=args.verbose_creation
        )
        print("Dataset creation completed.")

        if args.verbose_check:
            qds = QuadrupletDataset(os.path.join(args.test_out_path, args.dataset_name + "_test"),
                                    *[f"chunk_{i}.json" for i in range(0, 10)],
                                    hard_contrastive_mode=RANDOM,
                                    n_pos=1,
                                    n_neg=1,
                                    n_part_pos=1,
                                    cache_size=3)

            # 1st chunk
            print(qds[0])
            print(qds[3])
            print(qds[4])

            # 2nd chunk
            print(qds[qds.chunk_dim + 2])
            print(qds[qds.chunk_dim + 4])

            # 3rd chunk
            print(qds[qds.chunk_dim * 2])
            print(qds[qds.chunk_dim * 2 + 3])

            # 5th chunk
            print(qds[qds.chunk_dim * 4 + 1])
            print(qds[qds.chunk_dim * 4 + 5])

            # 6th chunk
            print(qds[qds.chunk_dim * 5 + 2])
            print(qds[qds.chunk_dim * 5 + 9])

            # 3rd chunk
            print(qds[qds.chunk_dim * 2 + 2])
            print(qds[qds.chunk_dim * 2 + 2])

            # 1st chunk
            print(qds[6])
            print(qds[2])

            # Get more than one sample
            print(qds[[1, 2, 3]])
    else:
        raise NotImplementedError(f"{SENTENCE_SUMMARIZATION_DS_TYPE} dataset creation is not supported yet.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_root', type=str, default=RAW_DATA,
                        help='raw data root (must contain an "images" empty subdir)')
    parser.add_argument('--dataset_name', type=str, default=COCO_DS_NAME, help='cleaned dataset folder name')
    parser.add_argument('--raw_train_path', type=str, default=COCO_CAPTIONS_TRAIN, help='train dataset path')
    parser.add_argument('--raw_val_path', type=str, default=COCO_CAPTIONS_VAL, help='val dataset path')
    parser.add_argument('--train_out_path', type=str, default=CLEANED_COCO_TRAIN, help='train set output path')
    parser.add_argument('--test_out_path', type=str, default=CLEANED_COCO_TEST, help='test set output path')
    parser.add_argument('--dataset_type', type=str, default=COCO_DS_TYPE,
                        choices=[COCO_DS_TYPE, SENTENCE_SUMMARIZATION_DS_TYPE],
                        help=f'dataset type, either {COCO_DS_TYPE} or {SENTENCE_SUMMARIZATION_DS_TYPE}')
    parser.add_argument('--verbose_check', type=bool, default=True,
                        help='whether to print random elements from created datasets for sanity checking')
    parser.add_argument('--verbose_creation', type=bool, default=True,
                        help='whether to log the progress of the dataset creation')
    parser.add_argument('--start_chunk_train', type=int, default=0,
                        help='the train set chunk from which to start the dataset creation (inclusive)')
    parser.add_argument('--last_chunk_train', type=int, default=-1,
                        help='the train set chunk where to stop the dataset creation (inclusive), -1 means last')
    parser.add_argument('--start_chunk_test', type=int, default=0,
                        help='the test set chunk from which to start the dataset creation (inclusive)')
    parser.add_argument('--last_chunk_test', type=int, default=-1,
                        help='the test set chunk where to stop the dataset creation (inclusive), -1 means last')
    parser.add_argument('--chunk_dim', type=int, default=CHUNK_DIM, help="number of instances per dataset chunk")
    parser.add_argument('--n_pos_examples', type=int, default=N_EXAMPLES,
                        help="number of positive examples per instance")
    parser.add_argument('--n_part_pos_examples', type=int, default=N_PART_EXAMPLES,
                        help="number of partially positive examples per instance")
    parser.add_argument('--pos_sim_threshold', type=float, default=SIMILARITY_THRESHOLD,
                        help='semantic similarity threshold for two instances to be considered a positive example')
    arguments = parser.parse_args()
    main(arguments)
