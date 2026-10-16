import pandas as pd
import numpy as np
import random


class PenguinDataLoader:
    def __init__(self, csv_path="penguins.csv"):
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        self.species_list = ["Adelie", "Chinstrap", "Gentoo"]
        self.feature_names = [
            "CulmenLength",
            "CulmenDepth",
            "FlipperLength",
            "OriginLocation",
            "BodyMass",
        ]
        self.load_and_preprocess()

    def load_and_preprocess(self):
        """Load CSV, handle missing values, encode OriginLocation."""
        self.df = pd.read_csv(self.csv_path)

        # 1. Fill missing numeric values with species-mean
        numeric_cols = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
        for col in numeric_cols:
            means = self.df.groupby("Species")[col].transform("mean")
            self.df[col] = self.df[col].fillna(means)

        # 2. Encode OriginLocation (Label Encoding)
        location_map = {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
        self.df["OriginLocation"] = self.df["OriginLocation"].map(location_map)

        if self.df.isnull().any().any():
            print("Warning: There are still missing values after preprocessing!")
            print(self.df.isnull().sum())

    def get_data(
        self, feature1, feature2, class1, class2, random_seed=42, normalize=True
    ):
        """
        Returns a dict with train/test splits (normalized + raw).

        Parameters
        ----------
        feature1, feature2 : str
            Names of the two features to use.
        class1, class2 : str
            Names of the two classes (class1 -> +1, class2 -> -1).
        random_seed : int
            Seed for the 30/20 train/test split.
        normalize : bool
            If True, apply z-score standardization (recommended).

        Returns
        -------
        dict with keys:
            X_train, y_train, X_test, y_test        # normalized (used by model)
            X_train_raw, X_test_raw                 # original scale (for plotting)
            mean, std                               # normalization stats
            feature_names                           # (f1, f2) for convenience
        """
        # ---- Validation ----
        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            raise ValueError(f"Features must be from {self.feature_names}")
        if class1 not in self.species_list or class2 not in self.species_list:
            raise ValueError(f"Classes must be from {self.species_list}")
        if class1 == class2:
            raise ValueError("The two classes must be different.")
        if feature1 == feature2:
            raise ValueError("The two features must be different.")

        # ---- Filter two classes ----
        df_filtered = self.df[self.df["Species"].isin([class1, class2])].copy()
        df_filtered["label"] = df_filtered["Species"].apply(
            lambda s: 1 if s == class1 else -1
        )

        data = df_filtered[[feature1, feature2, "label"]].values

        data_class1 = data[data[:, 2] == 1]
        data_class2 = data[data[:, 2] == -1]

        # ---- Stratified 30/20 split per class ----
        random.seed(random_seed)
        idx1 = list(range(len(data_class1)))
        idx2 = list(range(len(data_class2)))
        random.shuffle(idx1)
        random.shuffle(idx2)

        train_idx1, test_idx1 = idx1[:30], idx1[30:]
        train_idx2, test_idx2 = idx2[:30], idx2[30:]

        X_train = np.vstack(
            [
                data_class1[train_idx1][:, :2],
                data_class2[train_idx2][:, :2],
            ]
        )
        y_train = np.hstack(
            [
                data_class1[train_idx1][:, 2],
                data_class2[train_idx2][:, 2],
            ]
        )

        X_test = np.vstack(
            [
                data_class1[test_idx1][:, :2],
                data_class2[test_idx2][:, :2],
            ]
        )
        y_test = np.hstack(
            [
                data_class1[test_idx1][:, 2],
                data_class2[test_idx2][:, 2],
            ]
        )

        # Shuffle training rows
        perm = list(range(len(y_train)))
        random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]

        # ---- Keep raw copies for plotting ----
        X_train_raw = X_train.copy()
        X_test_raw = X_test.copy()

        # ---- Normalization (z-score) ----
        mean = std = None
        if normalize:
            mean = X_train_raw.mean(axis=0)
            std = X_train_raw.std(axis=0)
            std[std == 0] = 1.0  # avoid divide-by-zero

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
            "feature_names": (feature1, feature2),
        }


# Example usage / test
if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "..", "penguins.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, "penguins.csv")

    loader = PenguinDataLoader(csv_path)
    data = loader.get_data("FlipperLength", "BodyMass", "Adelie", "Gentoo")

    print("Train normalized shape:", data["X_train"].shape)
    print("Train raw shape:", data["X_train_raw"].shape)
    print("Mean:", data["mean"])
    print("Std :", data["std"])
    print("First normalized sample:", data["X_train"][0])
    print("First raw sample:", data["X_train_raw"][0])
