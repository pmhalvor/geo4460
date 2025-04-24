import dask
import time
from abc import ABC
from abc import abstractmethod

from dask.distributed import Client


class FeatureBase(ABC):
    def __init__(self, data_path):
        self.data_path = data_path

    @abstractmethod
    def interpolate(self):
        pass

    @dask.delayed
    def reclassify(self, category, feature_name):
        time.sleep(2)
        print(f"Reclassifying {category} data from {self.data_path} in {feature_name}")
        return f"Reclassified {category} data from {self.data_path}"

    @dask.delayed
    def buffer(self, feature_name):
        time.sleep(2)
        print(f"Buffering data from {self.data_path} in {feature_name}")
        return f"Buffered data from {self.data_path}"

    def load_data(self):
        # Simulate loading data wbt
        time.sleep(1)
        print(f"Loading data from {self.data_path}")
        return f"Loaded data from {self.data_path}"


class FeatureA(FeatureBase):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.load_data()
        self.interpolation_1_task = self.interpolate(1)
        self.interpolation_2_task = self.interpolate(2)
        print(f"FeatureA initialized with data path: {self.data_path}")

    @dask.delayed
    def interpolate(self, version):
        if version == 1:
            time.sleep(5)
            print(f"Interpolating data from {self.data_path} in FeatureA (version 1)")
        else:
            time.sleep(3)
            print(f"Interpolating data from {self.data_path} in FeatureA (version 2)")
        return f"Interpolated data from {self.data_path} (version {version})"


class FeatureB(FeatureBase):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.load_data()

        self.buffer_task = self.buffer("feature_b")
        self.interpolation_task = self.interpolate()
        self.reclassify_task = self.reclassify("category", "feature_b")
        print(f"FeatureB initialized with data path: {self.data_path}")

    @dask.delayed
    def interpolate(self):
        time.sleep(7)
        print(f"Interpolating data from {self.data_path} in FeatureB")
        return f"Interpolated data from {self.data_path}"


class FeatureC(FeatureBase):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.load_data()

        self.reclassify_task = self.reclassify("category", "feature_c")
        print(f"FeatureC initialized with data path: {self.data_path}")

    def interpolate(self):
        raise NotImplementedError("No interpolation defined for FeatureC")


def load_data(data_path):
    print("--- Start Data Loading ---")
    print(f"Loading data from {data_path}")

    client = Client()

    feature_a = FeatureA(data_path)
    feature_b = FeatureB(data_path)
    feature_c = FeatureC(data_path)

    tasks = [
        feature_a.interpolation_1_task,
        feature_a.interpolation_2_task,
        feature_b.buffer_task,
        feature_b.interpolation_task,
        feature_b.reclassify_task,
        feature_c.reclassify_task,
    ]

    results = dask.compute(*tasks)

    client.close()
    print("--- All tasks completed. ---")
    return results


if __name__ == "__main__":
    data_path = "path/to/data"
    print(load_data(data_path))
