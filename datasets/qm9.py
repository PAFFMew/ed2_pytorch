import os
import logging
import pickle
import shutil
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
from datasets.xtb import prepare_xtb_input, run_xtb


class QM9Dataset(Dataset):
    def __init__(
            self,
            qm9_src_dir: str,
            qm9_output_dir: str,
    ):
        super(QM9Dataset, self).__init__()

        self.src_dir = qm9_src_dir
        self.output_dir = qm9_output_dir

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def _read_qm9_file(self, path: str):
        """ Extract data from single xyz file in qm9 dataset.

        Example:
            An example of QM9 XYZ file:
            ================================================================================================================
            5
            gdb 1	157.7118	157.70997	157.70699	0.	13.21	-0.3877	0.1171	0.5048	35.3641	0.044749	-40.47893	-40.476062	-40.475117	-40.498597	6.469
            C	-0.0126981359	 1.0858041578	 0.0080009958	-0.535689
            H	 0.002150416	-0.0060313176	 0.0019761204	 0.133921
            H	 1.0117308433	 1.4637511618	 0.0002765748	 0.133922
            H	-0.540815069	 1.4475266138	-0.8766437152	 0.133923
            H	-0.5238136345	 1.4379326443	 0.9063972942	 0.133923
            1341.307	1341.3284	1341.365	1562.6731	1562.7453	3038.3205	3151.6034	3151.6788	3151.7078
            C	C
            InChI=1S/CH4/h1H4	InChI=1S/CH4/h1H4
            ================================================================================================================

        Note:
            I.  Property  Unit         Description
            --  --------  -----------  --------------
            1   tag       -            "gdb9"; string constant to ease extraction via grep
            2   index     -            Consecutive, 1-based integer identifier of molecule
            3   A         GHz          Rotational constant A
            4   B         GHz          Rotational constant B
            5   C         GHz          Rotational constant C
            6   mu        Debye        Dipole moment
            7   alpha     Bohr^3       Isotropic polarizability
            8   homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
            9   lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
            10  gap       Hartree      Gap, difference between LUMO and HOMO
            11  r2        Bohr^2       Electronic spatial extent
            12  zpve      Hartree      Zero point vibrational energy
            13  U0        Hartree      Internal energy at 0 K
            14  U         Hartree      Internal energy at 298.15 K
            15  H         Hartree      Enthalpy at 298.15 K
            16  G         Hartree      Free energy at 298.15 K
            17  Cv        cal/(mol K)  Heat capacity at 298.15 K

        Args:
            path (str): The path of single QM9 XYZ file.

        Returns:
            num_atoms (int): Number of atoms in molecule.
            prop_dict (Dict): Properties extracted from files.
            coords (np.ndarray): Atoms and coordinates.
            frequencies (np.ndarray): Molecular frequencies.
            smiles (str): Molecule SMILES string.
        """
        with open(path) as qm9_file:
            data = qm9_file.readlines()
        data = [line.strip('\n') for line in data]
        num_atoms = int(data[0])
        properties = data[1].split('\t')[:-1]  # End: ''
        dict_keys = ['tag_index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
                     'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        prop_dict = {}
        for idx, prop_name in enumerate(dict_keys):
            prop_dict[prop_name] = properties[idx]
        coords = data[2:2 + num_atoms]
        coords = [c.replace('*^', 'e') for c in coords]
        coords = [c.split('\t')[:-1] for c in coords]
        frequencies = data[2 + num_atoms].split('\t')
        smiles = ''.join(data[3 + num_atoms].split('\t')[0])
        # TODO (zhangtao): InChi string cannot be returned in source code,
        #                  maybe it is useless but I reserve it.
        inchi = data[4 + num_atoms]

        return num_atoms, prop_dict, coords, frequencies, smiles

    def _parse_single_qm9_smiles(
            self,
            qm9_file: str,
            rewrite: bool = True,
            esp: bool = True
    ):
        """Computes electron density, extracts properties and SMILES string
        from a qm9 xyz file. Results are stores in the pickle file.

        Args:
            qm9_file (str): a path to qm9 xyz file to process
            rewrite (bool): If true (default) it will redo the folder. If false, if
                        the folder exists, it won't do anything and leave it as it is.
            esp (bool): If using xtb to calculate electrostatic potentials

        Returns:
                None
        """
        # First prepare the path to the qm9 file, and prepare the output folder
        qm9_file_path = os.path.join(self.src_dir, qm9_file)
        file_id = qm9_file.strip('.xyz').split('_')[1]  # "dsgdb9nsd_000001.xyz" -> "000001"
        output_dir = os.path.join(self.output_dir, file_id)
        # FIXME (zhangtao):
        #  This logic is actually incorrect. Assuming there is an empty directory, the following `if`
        #  statement will terminate this method, but It's wrong because the necessary files do not exist.
        #  The correct logic should be: after confirming the existence of the directory, confirm whether all
        #  the necessary files in the directory exist. Of course, I will rewrite it.
        if os.path.exists(output_dir) and not rewrite:
            # logger.info(f'Folder {output_dir} exists. Not re-doing it.')
            return
        os.makedirs(output_dir, exist_ok=True)

        # read qm9 file, prepare and execute xtb
        num_atoms, properties, coords, _, smiles = self._read_qm9_file(qm9_file_path)
        xtb_input_file_path = os.path.join(output_dir, 'input.xtb')
        prepare_xtb_input(coords, xtb_input_file_path)
        output_dir = os.path.abspath(output_dir)
        xtb_exec_path = shutil.which('xtb')
        run_xtb(xtb_exec_path, xtb_input_file_path, output_dir, molden=True, esp=esp)

        # calculate electron density using orbkit, from xtb results
        molden_input = os.path.join(output_dir, 'molden.input')
        rho = electron_density_from_molden(molden_input, n_points=self.n_points,
                                           step_size=self.step_size)
        espxtb_input = os.path.join(output_dir, 'xtb_esp.dat')
        # calculate esp cube from xtb
        if esp:
            molecule_esp = self.esp.calculate_espcube_from_xtb(espxtb_input)

        output_dict = {}
        output_dict['electron_density'] = rho
        if esp:
            output_dict['electrostatic_potential'] = molecule_esp
        output_dict['properties'] = properties
        output_dict['smiles'] = smiles
        output_dict['num_atoms'] = int(num_atoms)
        output_file = os.path.join(output_dir, 'output.pkl')
        with open(output_file, 'wb+') as ofile:
            pickle.dump(output_dict, ofile)


if __name__ == '__main__':
    a, b, c, d, e = read_qm9_file('../data/dsgdb9nsd_000001.xyz')
