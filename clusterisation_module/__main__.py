
__author__ = 'mihailnikolaev'

from datetime import datetime
import numpy as np
import argparse
from user_processing_module import UserProcessing


# TODO: settings file for different clustering models
if __name__ == '__main__':
    start_time = datetime.now()
    print("Start time: {}".format(start_time))
    # Arguments for command line
    parser = argparse.ArgumentParser(description='Clustering module for data')
    parser.add_argument('--clusters', '-K', metavar='K', type=int, default=5, help='Number of desired clusters')
    parser.add_argument('--predict', '-p', metavar='p', type=bool, default=False, help='Trigger for data prediction')
    parser.add_argument('--file_path', '-f', metavar='f', type=str, default='../data/data.csv',
                        help='Path to file with data')
    parser.add_argument('--obj_file_path', '-obj', metavar='obj', type=str, default='proc_obj',
                        help='Path to serialized model file')
    parser.add_argument('--result_file_path', '-res', metavar='res', type=str, default='../data/results.csv',
                        help='Path to file with results')
    parser.add_argument('--method', '-m', metavar='M', type=str, default='DPGMM', help='Clustering method')
    args = vars(parser.parse_args())

    processing = UserProcessing(args['file_path'])  # Create new processing instance

    # if mode == predict from learned model do prediction
    if args['predict']:
        processing.deserialize(args['obj_file_path'])
        y = processing.predict(processing.data)
        data = np.hstack((processing.data, y[:, None]))
        print("Write results to file")
        processing.save_to_file(processing.describe_results(data, prdeiction_mode=args['predict']),
                                path=args['result_file_path'], mode='w')
        print("Results have been written")
    else:
        # if mode == learn fit the model and save it
        fitted_data = processing.fit_model(args['method'], clusters=args['clusters'])
        processing.serialize()
        print("Write results to file")
        processing.save_to_file(processing.describe_results(fitted_data), path=args['result_file_path'], mode='w')
        print("Results have been written")
    end_time = datetime.now()
    print("End time: {}".format(end_time))
    print("Operation duration {}".format(end_time - start_time))
    print("---------------------------------")