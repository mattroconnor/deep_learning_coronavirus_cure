from lstm_chem.utils.smiles_tokenizer2 import SmilesTokenizer
from lstm_chem.generator import LSTMChemGenerator


class LSTMChemFinetuner(LSTMChemGenerator):
    def __init__(self, modeler, finetune_data_loader):
        self.session = modeler.session
        self.model = modeler.model
        self.config = modeler.config
        self.finetune_data_loader = finetune_data_loader
        self.st = SmilesTokenizer()

    def finetune(self):
        self.model.compile(optimizer=self.config.optimizer,
                           loss='categorical_crossentropy')

        history = self.model.fit_generator(
            self.finetune_data_loader,
            steps_per_epoch=self.finetune_data_loader.__len__(),
            epochs=self.config.finetune_epochs,
            verbose=self.config.verbose_training,
            use_multiprocessing=True,
            shuffle=True)
        return history
