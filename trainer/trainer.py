import os
import pathlib

import torch
from model import model
from data_loader import data_loaders
from model import loss
from tqdm import tqdm
from model import metric
from logger import logger
from model import model_save
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Trainer():
    def __init__(self, batch_size: int = 64, loss_func: str = "CE", lr=0.0001, epochs: int = 100):
        self.BATCHSIZE = batch_size

        self.dataset = data_loaders.DogNCatDataSet()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, self.BATCHSIZE, shuffle=True)
        self.LEARNING_RATE = lr

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.model = model.DnCModel()
        self.model = model.AlexNet()
        self.model = self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE, weight_decay=0.005)
        self.criterion = loss.set_loss(loss_func)
        self.EPOCHS = epochs
        self.board_writer = logger.tensorboard_logger(pathlib.Path.cwd().parent, 'training_log')
        self.target_acc = 0.85
        print(f'current using device : {self.device}')
        print(f'Target Epochs:{self.EPOCHS}, Learning Rage:{self.LEARNING_RATE}, Using Model:{self.model._get_name()}')

    def _train(self):
        TOTAL = len(self.dataset)

        self.model.train()
        loss = 0
        for epoch in range(self.EPOCHS):
            acc = 0
            batch_total = 0
            with tqdm(self.dataloader) as pbar:
                for idx, data in enumerate(pbar):
                    if batch_total > 0:
                        pbar.set_description(
                            f'loss:{loss:.3f}, accuarcy:{acc}/{batch_total}, acc_rate:{acc / batch_total:.4f}')
                    self.optim.zero_grad()
                    x = data[0].to(self.device)
                    y = data[1].to(self.device)
                    z = self.model(x)
                    loss = self.criterion(z, y)
                    loss.backward()
                    self.optim.step()
                    acc += metric.acc_count(z, y)
                    batch_total += x.shape[0]

                self.board_writer.add_scalar('loss', loss, epoch)
                self.board_writer.add_scalar('accuracy', acc, epoch)
                for name, param in self.model.named_parameters():
                    self.board_writer.add_histogram(name, param.detach().cpu().data.numpy(), epoch)
                self.board_writer.add_histogram('sampleResult',
                                                torch.nn.functional.softmax(z, dim=1).detach().cpu().numpy(), epoch)
                self.board_writer.flush()
                print(f"EPOCH {epoch} : accuracy = {acc / TOTAL:.5f}, acc : {acc}, TOTAL : {TOTAL}")
                if acc/TOTAL>=self.target_acc:
                    model_save.save_model(self.model, epoch, acc/TOTAL)
                    self.target_acc += (1-self.target_acc)*0.1


    def _test(self):
        pass

    def start(self):
        self._train()


T = Trainer()
T.start()
