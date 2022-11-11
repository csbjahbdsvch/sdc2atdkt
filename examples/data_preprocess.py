import os, sys
import argparse
sys.path.append("../pykt")
from preprocess import process_raw_data

dname2paths = {
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "nips_task34": "../data/nips_task34/train_task_3_4.csv"
}
configf = "../configs/data_config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_name", type=str, default="algebra2005")
    parser.add_argument("-m","--min_seq_len", type=int, default=3)
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    args = parser.parse_args()

    print(args)

    from preprocess.split_datasets import main as split

    # process raw data
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-"*50)
    # split
    os.system("rm " + dname + "/*.pkl")

    split(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)
    print("="*100)
