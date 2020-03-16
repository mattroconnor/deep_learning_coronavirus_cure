import os
import time
from tensorflow.keras import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.initializers import RandomNormal
from lstm_chem.utils.smiles_tokenizer2 import SmilesTokenizer


class LSTMChem(object):
    def __init__(self, config, session='train'):
        assert session in ['train', 'generate', 'finetune'], \
                'one of {train, generate, finetune}'

        self.config = config
        self.session = session
        self.model = None

        if self.session == 'train':
            self.build_model()
        else:
            self.model = self.load(self.config.model_arch_filename,
                                   self.config.model_weight_filename)

    def build_model(self):
        st = SmilesTokenizer()
        n_table = len(st.table)
        weight_init = RandomNormal(mean=0.0,
                                   stddev=0.05,
                                   seed=self.config.seed)

        self.model = Sequential()
        self.model.add(
            LSTM(units=self.config.units,
                 input_shape=(None, n_table),
                 return_sequences=True,
                 kernel_initializer=weight_init,
                 dropout=0.3))
        self.model.add(
            LSTM(units=self.config.units,
                 input_shape=(None, n_table),
                 return_sequences=True,
                 kernel_initializer=weight_init,
                 dropout=0.5))
        self.model.add(
            Dense(units=n_table,
                  activation='softmax',
                  kernel_initializer=weight_init))

        arch = self.model.to_json(indent=2)
        self.config.model_arch_filename = os.path.join(self.config.exp_dir,
                                                       'model_arch.json')
        with open(self.config.model_arch_filename, 'w') as f:
            f.write(arch)

        self.model.compile(optimizer=self.config.optimizer,
                           loss='categorical_crossentropy')

    def save(self, checkpoint_path):
        assert self.model, 'You have to build the model first.'

        print('Saving model ...')
        self.model.save_weights(checkpoint_path)
        print('model saved.')

    def load(self, model_arch_file, checkpoint_file):
        print(f'Loading model architecture from {model_arch_file} ...')
        with open(model_arch_file) as f:
            model = model_from_json(f.read())
        print(f'Loading model checkpoint from {checkpoint_file} ...')
        model.load_weights(checkpoint_file)
        print('Loaded the Model.')
        return model
