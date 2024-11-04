from fastai.vision.all import *

from fastai.vision.models.xresnet import *
from fastai.distributed import *
from accelerate.utils import write_basic_config
from accelerate import notebook_launcher
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class TrainDentAge():
    def __init__(self, train_folder, test_folder, csv_path, used_model = resnet34, y_range = (5, 97), loss_func = L1LossFlat(), metrics = mae, bs = 64, num_workers = 0, save_path = 'xresnet34_save'):
        # defining the paths to images and labels
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.csv_path = csv_path
        self.save_path = save_path
        self.df = None

        # defining the dataset paths
        self.train_path = Path(train_folder)
        self.test_path = Path(test_folder)

        # training parameters
        self.used_model = used_model
        self.y_range = y_range
        self.loss_func = loss_func
        self.metrics = metrics
        self.bs = bs
        self.num_workers = num_workers

    def read_labels(self):
        # reading the labels dataframe
        self.df = pd.read_csv(self.csv_path, dtype=float, index_col=0)

    def read_images(self):
        # reading the folders looking for images
        self.train_files = get_image_files(self.train_path)
        self.test_files = sorted(get_image_files(self.test_path))
        # print the number of files found
        print(f'{len(self.train_files)} files were found for training and {len(self.test_files)} files were found for test')
        print(f'Age varies from {self.df.age.min()} to {self.df.age.max()}\n')
        _=self.df.age.hist()
    
    # function used to retun the label from an image


    def train(self):
        self.read_labels()
        self.read_images()

        df = self.df
        def label_func( file):
            # takes the file's base name w/o the extension
            basename = int(os.path.basename(str(file)).split('.')[0])
            # searches for the label 
            label = int(df.age[basename])
            return label

        # defines the image dataloader
        dtblk = DataBlock(blocks=(ImageBlock, RegressionBlock), 
                          get_items=get_image_files, 
                          get_y=label_func, 
                          item_tfms=Resize(200))
        
        dls = dtblk.dataloaders(self.train_folder, 
                                bs=self.bs, 
                                num_workers=self.num_workers)
        # shows some samples
        dls.show_batch()

        learn = vision_learner(dls, self.used_model, y_range=self.y_range, loss_func=self.loss_func, metrics=self.metrics)
        # training
        learn.fine_tune(epochs=1)
        learn.save(self.save_path, with_opt=False)

        print(f'{" "*26}PREDICTIONS:\n\n{" "*29}Label\n{" "*26}(predicted)')
        learn.show_results()

if __name__ == '__main__':
    train_folder = "path/to/train/images" #'Diamant/train'
    test_folder = "path/to/test/images" #'Diamant/test'
    csv_path = "path/to/csv/folder"# 'modified_labels.csv'

    my_class = TrainDentAge(train_folder, test_folder, csv_path)
    my_class.train()