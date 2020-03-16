#!/usr/bin/env python

import argparse
import os
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from lstm_chem.utils.smiles_tokenizer2 import SmilesTokenizer

RDLogger.DisableLog('rdApp.*')


class Preprocessor(object):
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def process(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normarizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None


def main(input_file, output_file, **kwargs):
    assert os.path.exists(input_file)
    assert not os.path.exists(output_file), f'{output_file} already exists.'
    print("kwargs? :", kwargs['finetune'])

    pp = Preprocessor()

    with open(input_file, 'r') as f:
        smiles = [l.rstrip() for l in f]

    print(f'input SMILES num: {len(smiles)}')
    print('start to clean up')

    pp_smiles = [pp.process(smi) for smi in tqdm(smiles)]
    print('Step 1 / 3 completed')
    cl_smiles = list(set([s for s in pp_smiles if s]))
    print('Step 2 / 3 completed')

    # token limits (34 to 128)
    out_smiles = []
    print('Initiating tokenizer')
    st = SmilesTokenizer()
    print('Tokenizer initiated')

    if kwargs['finetune']:
        print('In finetune kwargs')
        total = len(cl_smiles)
        print(total)
        count = 0
        skip_count = 0
        timeout_count = 0
        for cl_smi in cl_smiles:
            try:
                tokenized_smi = st.tokenize(cl_smi)
                if tokenized_smi == []:
                    timeout_count += 1     
                elif 34 <= len(tokenized_smi) <= 128:
                    out_smiles.append(cl_smi)          
            except:
                skip_count += 1
            count += 1
            if count % 25000 == 0:
                print(count, ' completed out of ', total,'. Skipped ', skip_count,'. Timed out ', timeout_count)
            # print(count, ' completed out of ', total,'. Have skipped ', skip_count)
    else:
        print('Not in finetune kwargs')
        out_smiles = cl_smiles

    print('done.')
    print(f'output SMILES num: {len(out_smiles)}')

    with open(output_file, 'w') as f:
        for smi in out_smiles:
            f.write(smi + '\n')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='remove salts and stereochemical infomation from SMILES')
    parser.add_argument('input', help='input file')
    parser.add_argument('output', help='output file')
    parser.add_argument('-ft',
                        '--finetune',
                        action='store_false',
                        help='for finetuning. ignore token length limitation.')
    args = parser.parse_args()
    main(args.input, args.output, finetune=args.finetune)
