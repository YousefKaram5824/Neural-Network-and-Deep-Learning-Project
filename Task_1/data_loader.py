import random

import numpy as np
import pandas as pd


class LoadingData:
    def __init__(self, csv_path="penguins.csv"):
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        self.species_list = ["Adelie", "Chinstrap", "Gentoo"]
        self.features_name = [
            "CulmenLength",
            "CulmenDepth",
            "FlipperLength",
            "OriginLocation",
            "BodyMass",
        ]
        self.load_and_preprocess()

    def load_and_preprocess(self):
        self.df = pd.read_csv(self.csv_path)

        num_columns = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
        for col in num_columns:
            means = self.df.groupby("Species")[col].transform("mean")
            self.df[col] = self.df[col].fillna(
                means
            )  # fill each species' numeric NaNs with its own species mean

        location = {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
        self.df["OriginLocation"] = self.df["OriginLocation"].map(
            location
        )  # encode string locations to 0/1/2

        if self.df.isnull().any().any():
            print("Warning: There are still missing values after preprocessing!")
            print(self.df.isnull().sum())

    def get_data(self, f1, f2, c1, c2, rand=42, normalize=True):
        if f1 not in self.features_name or f2 not in self.features_name:
            raise ValueError(f"Features must be from {self.features_name}")
        if c1 not in self.species_list or c2 not in self.species_list:
            raise ValueError(f"Classes must be from {self.species_list}")
        if c1 == c2:
            raise ValueError("The two classes must be different.")
        if f1 == f2:
            raise ValueError("The two features must be different.")

        df_filtered = self.df[self.df["Species"].isin([c1, c2])].copy()
        df_filtered["label"] = df_filtered["Species"].map({c1: 1, c2: -1})

        data = df_filtered[[f1, f2, "label"]].values
        data_class1 = data[data[:, 2] == 1]
        data_class2 = data[data[:, 2] == -1]

        random.seed(rand)
        idx1 = list(range(len(data_class1)))
        idx2 = list(range(len(data_class2)))
        random.shuffle(idx1)
        random.shuffle(idx2)

        train_idx1, test_idx1 = idx1[:30], idx1[30:]
        train_idx2, test_idx2 = idx2[:30], idx2[30:]

        tr1, te1 = data_class1[train_idx1], data_class1[test_idx1]
        tr2, te2 = data_class2[train_idx2], data_class2[test_idx2]

        X_train = np.vstack([tr1[:, :2], tr2[:, :2]])
        y_train = np.hstack([tr1[:, 2], tr2[:, 2]])
        X_test = np.vstack([te1[:, :2], te2[:, :2]])
        y_test = np.hstack([te1[:, 2], te2[:, 2]])

        perm = list(range(len(y_train)))
        random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]

        X_train_raw = X_train.copy()
        X_test_raw = X_test.copy()

        mean = std = None
        if normalize:  # normalize using training-set statistics only
            mean = X_train_raw.mean(axis=0)
            std = X_train_raw.std(axis=0)
            std[std == 0] = 1.0
            X_train = (X_train_raw - mean) / std
            X_test = (X_test_raw - mean) / std

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "mean": mean,
            "std": std,
            "feature_names": (f1, f2),
        }
