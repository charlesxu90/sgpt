import os
import time
import sys
import math
import tempfile
import shutil
import logging
import subprocess
import multiprocessing
import pandas as pd
from copy import deepcopy
from shlex import quote
from pathlib import Path
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from enum import Enum
from rdkit import Chem
from agent.docker.ligand import Ligand

"""
Demo example:
1. save smiles to pdf format in python
```python
import rdkit.Chem as Chem

smiles = 'C#CCCCn1c(Cc2cc(OC)c(OC)c(OC)c2Cl)nc2c(N)ncnc21'
molecule = Chem.MolFromSmiles(smiles)
pdb_path = 'data/vina/ligand.pdb'
Chem.MolToPDBFile(mol=molecule, filename=pdb_path)
```
2. convert pdb to pdbqt format
```shell
obabel -ipdb ligand.pdb -opdbqt -O ligand.pdbqt --partialcharge gasteiger
```
3. run docking
```shell
vina --receptor data/vina/ADV_receptor.pdbqt --ligand ligand.pdbqt --cpu 100 --seed 42 --out ligand_docked.pdbqt \
    --center_x 3.3 --center_y 11.5 --center_z 24.8 --size_x 15.0 --size_y 10.0 --size_z 10.0 --num_modes 2
```
4. convert result pdbqt file to sdf
```shell
obabel -ipdbqt ligand_docked.pdbqt -osdf -O ligand_docked.sdf
```
5. parse sdf file with python
"""

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger.addHandler(logging.NullHandler())


class Parallelization(BaseModel):
    n_jobs: Optional[int] = Field(default=4)
    n_ligands_per_job: Optional[int] = Field(default=1, ge=0)
    n_cpu_per_job: int = 4  # number of cpus per sub-job, i.e. one ligand


class SearchSpace(BaseModel):
    center_x: float = Field(alias="--center_x")
    center_y: float = Field(alias="--center_y")
    center_z: float = Field(alias="--center_z")
    size_x: float = Field(alias="--size_x")
    size_y: float = Field(alias="--size_y")
    size_z: float = Field(alias="--size_z")

    class Config:
        allow_population_by_field_name = True


class AutodockVinaParameters(BaseModel):
    binary_location: Optional[str] = None
    parallelization: Optional[Parallelization]
    receptor_pdbqt_path: Optional[List[str]] = None
    search_space: SearchSpace
    seed: int = 42
    number_poses: int = 1

    def get(self, key: str) -> Any:
        """Temporary method to support nested_get"""
        return self.dict()[key]


class OutputMode(str, Enum):
    all = "all"
    best_per_ligand = "best_per_ligand"
    best_per_enumeration = "best_per_enumeration"


class Output(BaseModel):
    poses_path: str
    scores_path: str
    mode: str = OutputMode.best_per_ligand


class Executor:
    def __init__(self, binary_location=None):
        self._binary_location = binary_location

    def _execute(self, command: str, arguments: list, check=True, location=None):
        arguments = [quote(str(arg)) for arg in arguments]  # wrap arguments to avoid errors
        if self._binary_location is not None:
            command = os.path.join(self._binary_location, command)
        complete_command = [command + ' ' + ' '.join(str(e) for e in arguments)]

        old_cwd = os.getcwd()  # save old dir and come back before return
        if location is not None:
            os.chdir(location)
        result = subprocess.run(complete_command, check=check, universal_newlines=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=True)
        os.chdir(old_cwd)
        return result


class AutodockVinaExecutor(Executor):

    def __init__(self, binary_location=None):
        super().__init__(binary_location=binary_location)

    def is_available(self):
        try:
            result = self.execute(command="vina", arguments=["--version"], check=True)

            if "AutoDock Vina 1.1.2" in result.stdout:  # check if vina is correct
                return True
            return False
        except Exception as e:
            return False

    def execute(self, command: str, arguments: list, check=True, location=None):
        # check, whether a proper executable is provided
        if command not in ["vina"]:
            raise ValueError("Command must be 'vina'.")

        return self._execute(command=command, arguments=arguments, check=check, location=None)


class OpenBabelExecutor(Executor):
    """For the execution of the "obabel" binary."""

    def __init__(self):
        super().__init__()

    def is_available(self):
        # unfortunately, "obabel" does not return a meaningful return value (always '1'), so instead try to parse
        # the "stdout" of the standard message; note, that "OpenBabel" is part of the environment and should always work
        try:
            result = self.execute(command="obabel", arguments=[], check=False)
            if "-O<outfilename>" in result.stdout:
                return True
            return False
        except Exception as e:
            return False

    def execute(self, command: str, arguments: list, check=True, location=None):
        # check, whether a proper executable is provided
        if command not in ["obabel"]:
            raise ValueError("Command must be 'obabel'.")

        return self._execute(command=command, arguments=arguments, check=check, location=location)


class AutodockResultParser:
    """Class that loads, parses and analyzes the output of an "AutoDock Vina" docking run,
        including poses and scores."""
    def __init__(self, ligands):
        self._ligands = ligands
        self._df_results = self._construct_dataframe()

    def _construct_dataframe(self) -> pd.DataFrame:
        data_buffer = []
        for ligand in self._ligands:
            best = True
            for conformer_index, conformer in enumerate(ligand.get_conformers()):
                name = self._get_name(ligand, conformer_index)
                row = [ligand.get_ligand_number(), conformer_index, name,
                       float(conformer.GetProp("SCORE")), ligand.get_smile(), best]
                best = False
                data_buffer.append(row)
        return pd.DataFrame(data_buffer, columns=["ligand_number", "conformer_number", "name", "score",
                                                  "smiles", "lowest_conformer"])

    @staticmethod
    def _get_name(ligand: Ligand, conformer_index: int):
        """Get either the name (for named molecules) or the identifier (plus the conformer)."""
        if ligand.get_name() is None:
            return ligand.get_identifier() + ':' + str(conformer_index)
        else:
            return ligand.get_name()

    def as_dataframe(self):
        return deepcopy(self._df_results)


class AutodockVina(BaseModel):
    """AutoDock Vina Docking"""
    parameters: AutodockVinaParameters
    output: Optional[Output]
    ligands: List = []

    _docking_performed = False
    _df_results = None

    # executors to run commands in command line
    _ADV_executor: AutodockVinaExecutor = None
    _OpenBabel_executor: OpenBabelExecutor = None

    class Config:
        underscore_attrs_are_private = True

    def __init__(self, **params):
        super().__init__(**params)
        self.parameters.binary_location = os.path.expanduser(self.parameters.binary_location)
        self.parameters.receptor_pdbqt_path = [os.path.expanduser(path) for path in self.parameters.receptor_pdbqt_path]
        self.output.poses_path = os.path.expanduser(self.output.poses_path)
        self.output.scores_path = os.path.expanduser(self.output.scores_path)

    def add_molecules(self, molecules: list):
        self.ligands = molecules
        self._docking_performed = False

    def dock(self):
        if len(self.ligands) == 0:
            raise Exception("No molecules to dock.")

        [ligand.clear_conformers() for ligand in self.ligands] # delete conformers

        try:
            n_jobs = self.parameters.parallelization.n_jobs
        except:
            n_jobs = 1

        if n_jobs < 0:  # use all cores except number_cores
            n_jobs = multiprocessing.cpu_count() + n_jobs

        self._dock(n_jobs=n_jobs)

    def _dock(self, n_jobs):
        self._initialize_executors()

        start_indices, sublists = self.get_sublists_for_docking(n_jobs=n_jobs, enforce_singletons=True)
        number_sublists = len(sublists)
        logger.debug(f"Ligands to dock {self.ligands}.")

        sublists_submitted = 0
        slices_per_iteration = min(n_jobs, number_sublists)
        while sublists_submitted < len(sublists):
            upper_bound_slice = min((sublists_submitted + slices_per_iteration), len(sublists))
            cur_slice_start_indices = start_indices[sublists_submitted:upper_bound_slice]
            cur_slice_sublists = sublists[sublists_submitted:upper_bound_slice]

            # generate temp paths and initialize molecules
            tmp_output_dirs, tmp_input_paths, tmp_output_paths, ligand_identifiers = \
                self._generate_temporary_input_output_files(cur_slice_start_indices, cur_slice_sublists)

            processes = []
            for chunk_index in range(len(tmp_output_dirs)):  # run in parallel
                p = multiprocessing.Process(target=self._dock_subjob, args=(tmp_input_paths[chunk_index],
                                                                            tmp_output_paths[chunk_index]))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            # Count number of smiles submitted, some with no molecule and not really running
            sublists_submitted += len(cur_slice_sublists)

            # parse the resulting sdf files and add conformers
            for path_sdf_results, cur_identifier in zip(tmp_output_paths, ligand_identifiers):
                if not os.path.isfile(path_sdf_results) or os.path.getsize(path_sdf_results) == 0:
                    continue

                for molecule in Chem.SDMolSupplier(path_sdf_results, removeHs=False):
                    if molecule is None:
                        continue

                    # extract the score from the AutoDock Vina output and update some tags
                    score = self._extract_score_from_vina(molecule=molecule)
                    logger.info(score)
                    molecule.SetProp("_Name", cur_identifier)
                    molecule.SetProp("SCORE", score)
                    molecule.ClearProp("REMARK")

                    # add molecule to the appropriate ligand
                    for ligand in self.ligands:
                        if ligand.get_identifier() == cur_identifier:
                            ligand.add_conformer(molecule)
                            break

            for path in tmp_output_dirs:  # clean-up
                shutil.rmtree(path)
            logger.info(self.get_progress_bar_string(sublists_submitted, number_sublists, length=65))

        for ligand in self.ligands:
            ligand.add_tags_to_conformers()

        logger.debug(f"Ligands docked {[lig for lig in self.ligands if len(lig.get_conformers()) > 0]}.")

        self._docking_fail_check()

        result_parser = AutodockResultParser(ligands=[ligand.get_clone() for ligand in self.ligands])
        self._df_results = result_parser.as_dataframe()
        self._docking_performed = True

    def _initialize_executors(self):
        """Initialize vina and openbabel executors."""
        self._ADV_executor = AutodockVinaExecutor(
            binary_location=self.parameters.binary_location
        )
        if not self._ADV_executor.is_available():
            raise Exception("Cannot initialize AutoDock Vina docker, as backend is not available - abort.")
        logger.debug(f"Checked AutoDock Vina backend availability.")

        self._OpenBabel_executor = OpenBabelExecutor()  # use command-line openbabel
        if not self._OpenBabel_executor.is_available():
            raise Exception("Cannot initialize OpenBabel. Please install it.")

    def get_sublists_for_docking(self, n_jobs=7, enforce_singletons=True):
        """Splits ligands into sublists for docking to take advantage of parallel computing.
        :param n_jobs: Number of cores to allocate docking jobs.
        :param enforce_singletons: Each sublist only contains 1 ligand. Necessary for AutoDock Vina, default True
        :return: list of sublists
        """
        if enforce_singletons:  # each sublist should have exactly one member, for AutoDock Vina
            return self.split_into_sublists(partitions=None, slice_size=1)

        try:
            ligands_per_job = self.parameters.parallelization.n_ligands_per_job
        except:
            ligands_per_job = 0

        if ligands_per_job > 0:
            slice_size = min(ligands_per_job, len(self.ligands))
            return self.split_into_sublists(partitions=None, slice_size=slice_size)
        else:
            partitions = min(n_jobs, len(self.ligands))  # split to as many cores as available
            return self.split_into_sublists(partitions=partitions, slice_size=None)

    def split_into_sublists(self, partitions=None, slice_size=None):
        if partitions is None and slice_size is None:
            raise ValueError("Either specify partitions or slice size.")

        return_list, start_indices = [], []  # store the index

        len_input = len(self.ligands)
        chunk_size = int(math.ceil(len_input / partitions)) if partitions is not None else slice_size

        for i in range(0, len_input, chunk_size):
            start_indices.append(i)
            return_list.append(self.ligands[i:i + chunk_size])
        return start_indices, return_list

    @staticmethod
    def get_progress_bar_string(done, total, prefix="", suffix="", decimals=1, length=100, fill='█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (done / float(total)))
        len_filled = int(length * done // total)
        bar = fill * len_filled + '-' * (length - len_filled)
        return f"{prefix}|{bar}| {percent}% {suffix}"

    @staticmethod
    def gen_temp_file(suffix=None, prefix=None, dir=None, text=True) -> str:
        filehandler, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
        os.close(filehandler)
        return path

    def _generate_temporary_input_output_files(self, start_indices, sublists):
        if not isinstance(start_indices, list):
            start_indices = [start_indices]
        if not isinstance(sublists, list):
            sublists = [sublists]

        tmp_output_dirs = []
        tmp_input_paths = []
        tmp_output_paths = []
        ligand_identifiers = []
        for start_index, sublist in zip(start_indices, sublists):
            for ligand in sublist:  # In "AutoDock Vina", only single molecule is handled in each sublist
                cur_tmp_output_dir = tempfile.mkdtemp()
                cur_tmp_input_pdbqt = self.gen_temp_file(prefix=str(start_index), suffix=".pdbqt", dir=cur_tmp_output_dir)
                cur_tmp_output_sdf = self.gen_temp_file(prefix=str(start_index), suffix=".sdf", dir=cur_tmp_output_dir)

                if ligand.get_molecule() is None:  # remove tmp dir if no molecule for this ligand
                    if os.path.isdir(cur_tmp_output_dir):
                        shutil.rmtree(cur_tmp_output_dir)
                    continue

                mol = deepcopy(ligand.get_molecule())
                self._write_molecule_to_pdbqt(cur_tmp_input_pdbqt, mol)  # write to pdbqt file
                tmp_output_dirs.append(cur_tmp_output_dir)
                tmp_input_paths.append(cur_tmp_input_pdbqt)
                tmp_output_paths.append(cur_tmp_output_sdf)
                ligand_identifiers.append(ligand.get_identifier())

        return tmp_output_dirs, tmp_input_paths, tmp_output_paths, ligand_identifiers

    def _write_molecule_to_pdbqt(self, path, molecule) -> bool:
        temp_pdb = self.gen_temp_file(suffix=".pdb")  # first, create a pdb file
        Chem.MolToPDBFile(mol=molecule, filename=temp_pdb)

        # transform pdb format to pdbqt with obabel
        arguments = [temp_pdb, '-opdbqt', "".join(["-O", path]), '--partialcharge', 'gasteiger']
        self._OpenBabel_executor.execute(command="obabel", arguments=arguments, check=False)

        return True if os.path.exists(path) else False

    def _dock_subjob(self, input_path_pdbqt, output_path_sdf):
        def delay4file_system(path, interval_sec=1, maximum_sec=10) -> bool:
            counter = 0
            while not os.path.exists(path):
                time.sleep(interval_sec)  # wait
                counter = counter + 1

                if maximum_sec is not None and counter * interval_sec >= maximum_sec:
                    break   # stop waiting if maximum_seq reached

            return True if os.path.exists(path) else False

        # Run docking
        tmp_docked_pdbqt = self.gen_temp_file(suffix=".pdbqt", dir=os.path.dirname(input_path_pdbqt))
        search_space = self.parameters.search_space
        arguments = ["--receptor", self.parameters.receptor_pdbqt_path[0],
                     "--ligand", input_path_pdbqt,
                     "--cpu", str(self.parameters.parallelization.n_cpu_per_job),
                     "--seed", self.parameters.seed,
                     "--out", tmp_docked_pdbqt,
                     "--center_x", str(search_space.center_x),
                     "--center_y", str(search_space.center_y),
                     "--center_z", str(search_space.center_z),
                     "--size_x", str(search_space.size_x),
                     "--size_y", str(search_space.size_y),
                     "--size_z", str(search_space.size_z),
                     "--num_modes", self.parameters.number_poses]
        execution_result = self._ADV_executor.execute(command="vina", arguments=arguments, check=True)
        delay4file_system(path=tmp_docked_pdbqt)

        # translate the output from PDBQT to SDF
        arguments = [tmp_docked_pdbqt, "-ipdbqt", "-osdf", "".join(["-O", output_path_sdf])]
        self._OpenBabel_executor.execute(command="obabel", arguments=arguments, check=False)
        delay4file_system(path=output_path_sdf)

    def _docking_fail_check(self):
        logger.debug(f"Attempted to dock {len(self.ligands)} molecules.")
        not_docked = len([lig for lig in self.ligands if len(lig.get_conformers()) == 0])
        logger.debug(f"{not_docked} ligand(s) failed to dock.")

    @staticmethod
    def _extract_score_from_vina(molecule) -> str:
        result_tag_lines = molecule.GetProp("REMARK").split("\n")
        result_line = [line for line in result_tag_lines if "VINA RESULT" in line][0]
        cols = result_line.split()
        return cols[2]  # based column after split

    def write_docked_ligands(self):
        path = self.output.poses_path

        if not self._docking_performed:
            raise Exception("Do the docking first.")
        selected_conformers = self._select_conformers()
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)  # create path if not available

        writer = Chem.SDWriter(path)
        for conformer in selected_conformers:
            writer.write(conformer)
        writer.close()
        logger.debug(f"Wrote docked ligands to file {path}.")

    def _select_conformers(self):
        mode = self.output.mode
        ligands = [deepcopy(lig) for lig in self.ligands]
        selected_conformers = [conformer for ligand in ligands for conformer in ligand.get_conformers()]  # all cons

        if mode == "all":
            return selected_conformers
        elif mode == "best_per_enumeration":
            selected_conformers = [con for con in selected_conformers if con.GetProp("_Name").endswith(":0")]
        elif mode == "best_per_ligand":  # filter down to "best_per_ligand"
            lig_ids = list(set([ligand.get_ligand_number() for ligand in ligands]))
            selected_conformers = self._get_best_con_per_ligand(conformers=selected_conformers, ligand_ids=lig_ids)
        return selected_conformers

    @staticmethod
    def _get_best_con_per_ligand(conformers: list, ligand_ids: list) -> list:
        con_grouped = {str(ligand_id): [] for ligand_id in ligand_ids}  # initialize dict: lig_id -> List(conformer)
        for conformer in conformers:
            cur_id = conformer.GetProp("_Name").split(':')[0]
            con_grouped[str(cur_id)].append(conformer)

        selected_conformers = []
        for ligand_id in ligand_ids:
            list_conf = con_grouped[str(ligand_id)]
            if len(list_conf) > 0:
                list_conf = sorted(conformers, key=lambda con: float(con.GetProp("SCORE")))
                selected_conformers.append(list_conf[0])
        return selected_conformers

    def write_result(self):
        def _write_out_buffer(path, buffer_df):
            Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)  # make path if not exist
            buffer_df.to_csv(path_or_buf=path, sep=',', na_rep='', header=True, index=False, mode='w', quoting=None)
            logger.info(f"Wrote docking results to file {path} ({buffer_df.shape[0]} rows).")

        if self._df_results is None or self._df_results.empty:
            logger.error("Result dataframe is empty, skipping write-out.")
            return

        df_buffer = self._df_results.copy()
        if self.output.mode == "all":
            return _write_out_buffer(self.output.scores_path, df_buffer)

        df_buffer = df_buffer[df_buffer["lowest_conformer"]]  # Where 'lowest_conformer' is True
        if self.output.mode == "best_per_enumeration":
            return _write_out_buffer(self.output.scores_path, df_buffer)

        df_buffer = df_buffer.loc[df_buffer.groupby("ligand_number")["score"].idxmin()]  # Score is the smallest
        if self.output.mode == "best_per_ligand":
            return _write_out_buffer(self.output.scores_path, df_buffer)

        logger.error(f"Output mode must be \"all\", \"best_per_enumeration\", or \"best_per_ligand\".")
        raise Exception()

    def get_scores(self, best_only=True):
        if not self._docking_performed:
            raise Exception("Do the docking first.")

        ligand_numbers = list(set([ligand.get_ligand_number() for ligand in self.ligands]))

        lig_num2scores = {lig_num: [] for lig_num in ligand_numbers}
        [lig_num2scores[ligand.get_ligand_number()].append(float(con.GetProp("SCORE")))
         for ligand in self.ligands for con in ligand.get_conformers()]  # add all conformer scores to lig_num2scores

        result_list = []
        for lig_num in ligand_numbers:
            if len(lig_num2scores[lig_num]) == 0:
                result_list.append("NA")
                continue

            if best_only:
                result_list.append(min(lig_num2scores[lig_num]))  # only add minimum
            else:
                result_list = result_list + lig_num2scores[lig_num]  # add all
        return result_list
