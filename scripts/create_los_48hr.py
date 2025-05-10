#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create data for a 3-class remaining LOS classification task based on the first 48 hours in ICU.
Classes (remaining stay after 48h):
  0: 48-96 hours (2-4 days)
  1: 96-144 hours (4-6 days)
  2: 144+ hours (6+ days)
Writes per-partition CSVs of truncated timeseries and a listfile.csv with class labels (0,1,2).
"""
from __future__ import absolute_import, print_function

import os
import argparse
import pandas as pd
import random
from tqdm import tqdm

random.seed(49297)


def process_partition(args, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xy_pairs = []
    root_part = os.path.join(args.root_path, partition)
    patients = [p for p in os.listdir(root_part) if p.isdigit()]

    for patient in tqdm(patients, desc=f'Iterating over {partition} patients'):
        patient_folder = os.path.join(root_part, patient)
        ts_files = [f for f in os.listdir(patient_folder) if "timeseries" in f]

        for ts_filename in ts_files:
            ts_path = os.path.join(patient_folder, ts_filename)
            lb_filename = ts_filename.replace("_timeseries", "")
            lb_path = os.path.join(patient_folder, lb_filename)

            # Read label file
            try:
                label_df = pd.read_csv(lb_path)
            except pd.errors.EmptyDataError:
                print(f"(empty label file) {patient} {lb_filename}")
                continue

            if label_df.empty or pd.isnull(label_df.loc[0, 'Length of Stay']):
                print(f"(empty or invalid label file) {patient} {lb_filename}")
                continue

            icustay_id = label_df.loc[0, 'Icustay']
            full_los_hrs = 24.0 * label_df.loc[0, 'Length of Stay']

            # Skip stays shorter than the observation window
            if full_los_hrs < n_hours - eps:
                continue

            # Compute remaining hours & bucket into classes
            remaining_hrs = max(full_los_hrs - n_hours, 0.0)
            if remaining_hrs <= 48.0:
                los_cls = 0
            elif remaining_hrs <= 96.0:
                los_cls = 1
            else:
                los_cls = 2

            # Read and truncate timeseries to first n_hours
            with open(ts_path, 'r') as tsfile:
                lines = tsfile.readlines()
            header, body = lines[0], lines[1:]
            times = [float(line.split(',')[0]) for line in body]
            truncated = [l for l, t in zip(body, times) if -eps < t < n_hours + eps]
            
            if not truncated:
                # no data in the first window
                print(f"(no events in first {n_hours} hours) {patient} {ts_filename}")
                continue

            # Write truncated timeseries
            out_ts_name = f"{patient}_{ts_filename}"
            with open(os.path.join(output_dir, out_ts_name), 'w') as out:
                out.write(header)
                out.writelines(truncated)

            xy_pairs.append((out_ts_name, icustay_id, los_cls))

    print(f"{partition}: created {len(xy_pairs)} samples")

    # Shuffle or sort
    if partition == "train":
        random.shuffle(xy_pairs)
    else:  # test
        xy_pairs.sort()

    # Write listfile.csv
    listfile_path = os.path.join(output_dir, "listfile.csv")
    with open(listfile_path, 'w') as listfile:
        listfile.write('stay,period_length,stay_id,y_true\n')
        for fn, stay_id, cls in xy_pairs:
            listfile.write(f"{fn},0,{stay_id},{cls}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create truncated timeseries + 3-class remaining LOS labels"
    )
    parser.add_argument('root_path', type=str,
                        help="Path to root folder containing 'train' and 'test' subfolders")
    parser.add_argument('output_path', type=str,
                        help="Directory where the processed data will be stored")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()