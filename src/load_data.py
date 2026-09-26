import pandas as pd


def load_source1(path):
    return pd.read_csv(path, sep="\t")


def load_source2(path):
    return pd.read_csv(path, sep="\t")


def load_source3(path):
    return pd.read_csv(path, sep="\t")


def load_ground_truth(path):
    return pd.read_csv(path, sep="\t")


if __name__ == "__main__":

    s1 = load_source1("dataset/train/train_source1.tsv")
    s2 = load_source2("dataset/train/train_source2.tsv")
    s3 = load_source3("dataset/train/train_source3.tsv")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")

    print("Source 1 shape:", s1.shape)
    print("Source 2 shape:", s2.shape)
    print("Source 3 shape:", s3.shape)
    print("Ground truth shape:", gt.shape)

    print("\nSource 1 columns:")
    print(s1.columns.tolist())

    print("\nSource 2 columns:")
    print(s2.columns.tolist())

    print("\nSource 3 columns:")
    print(s3.columns.tolist())

    print("\nGround truth columns:")
    print(gt.columns.tolist())

    print("\nSource 1 sample:")
    print(s1.head())

    print("\nSource 2 sample:")
    print(s2.head())

    print("\nSource 3 sample:")
    print(s3.head())

    print("\nGround truth sample:")
    print(gt.head())
