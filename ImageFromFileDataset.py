from torch.utils.data import Dataset
import os
from PIL import Image

class ImageFromFileDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_df, labels, class_names, root_dir, transform=None):
        """
        Arguments:
            file_df (string): Pandas DataFrame (2 columns: file_name, class_label)
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.file_df = pd.read_csv(csv_file)
        self.file_df = file_df
        self.labels = labels
        self.class_names = class_names
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_df)

    def __getitem__(self, idx):
        #print(idx)
        label = self.labels[idx]
        #print(label)
        class_name = self.class_names[label]
        img_name = os.path.join(self.root_dir, class_name,
                                self.file_df.iloc[idx])
        #print(img_name)
        #image = io.imread(img_name)
        image = Image.open(img_name)

        # Apply custom transformation if any
        if self.transform:
            image = self.transform(image)

        return image, label
