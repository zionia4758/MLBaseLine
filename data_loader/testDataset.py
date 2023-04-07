from base.base_data_loader import BaseDataLoader

class testDataset(BaseDataLoader):
    def __init__(self,base_dir,data,annotation,shuffle=None,valid=None,trans=None):
        super().__init__('../data','images','annotation')
        print("test")

    def getData(self,data_dir):
        data_list = data_dir
        print("get data",data_dir)
        return data_list
    def getAnnotation(self,annotation_dir):
        annotation_list = annotation_dir
        print("getanno",annotation_dir)
        return annotation_list

    def __len__(self):
        return self.data_list
    def __getitem__(self,idx):
        return

testDataset("1","2","3")