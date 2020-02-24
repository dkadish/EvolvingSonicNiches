import os, typing

import joblib


def load_subset_fn(f: typing.Callable, dir: str, filename: str = 'data.joblib'):
    data = []
    for d in load_runset(dir, filename):
        data.append(f(d))

    return data


def load_subset(key: str, dir: str, filename: str = 'data.joblib'):
    data = []
    for d in load_runset(dir, filename):
        data.append(d[key])

    return data


def load_runset(dir: str, filename: str = 'data.joblib'):
    datafiles = []
    # Load data files
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.strip() == filename.strip():
                print('{}: {}'.format(len(datafiles), root))
                datafiles.append(os.path.join(root, file))

    data = []
    for df in datafiles:
        data.append(joblib.load(df))

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test graphing functions.')
    parser.add_argument('dir', help='Folder to search within for the data files.')
    parser.add_argument('datafile', help='File containing joblibed run data.')
