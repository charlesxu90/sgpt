import os
from copy import deepcopy
import logging
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, PrivateAttr
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from agent.docker.ligand import Ligand


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger.addHandler(logging.NullHandler())


class CoordinateGenerationParams(BaseModel):
    method: str = "UFF"
    maximum_iterations: Optional[int] = 600


class RDkitLigPrepParams(BaseModel):
    protonate: Optional[bool] = True
    coordinate_generation: CoordinateGenerationParams = CoordinateGenerationParams()


class Input(BaseModel):
    type: Optional[str] = "smi"
    input_path: Optional[str]


class Output(BaseModel):
    format: str = "sdf"
    conformer_path: str


class RDkitLigandPreparator(BaseModel):
    """Class to prepare ligand with RDkit"""
    input: Input
    output: Optional[Output]
    parameters: RDkitLigPrepParams = RDkitLigPrepParams()

    ligands: Optional[List] = None
    _references: List = PrivateAttr(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        if self.ligands is not None and len(self.ligands) >= 1:  # check ligand error
            if not isinstance(self.ligands, list):
                self.ligands = [self.ligands]
            if len(self.ligands) == 0:
                raise Exception("Specify at least one ligand (or a list).")

        self.input.input_path = os.path.expanduser(self.input.input_path)  # enable ~ in path
        self.output.conformer_path = os.path.expanduser(self.output.conformer_path)

    def get_ligands(self):
        return self.ligands
    
    @staticmethod
    def _smiles_to_molecules(ligands: List[Ligand]) -> List[Ligand]:
        """Get ligands and set molecules"""
        for lig in ligands:
            smi = lig.get_smile()
            if smi:
                mol = Chem.MolFromSmiles(smi)
                lig.set_molecule(mol)
        return ligands

    def generate_3d_coordinates(self, converged_only=False):
        """Method to generate 3D coordinates, in case the molecules have been built from SMILES."""
        RDLogger.DisableLog("rdApp.*")
        ligand_list = self._smiles_to_molecules(deepcopy(self.ligands))

        failed = 0
        succeeded = 0
        for idx, lig_obj in enumerate(ligand_list):
            ligand = lig_obj.get_molecule()
            if ligand is None:
                continue

            # "useRandomCoords" set to "True" to handle embedding fails of larger molecules
            try:
                embed_code = AllChem.EmbedMolecule(ligand, randomSeed=42, useRandomCoords=True)
            except:
                embed_code = -1

            if embed_code == -1:  # Embed error, not only except
                logger.debug(f"Could not embed molecule number {lig_obj.get_ligand_number()} " +
                             f"(smile: {lig_obj.get_smile()}) - no 3D coordinates generated.")
                failed += 1
                continue

            # use UFF rather than MMFF, as UFF fail less often and is much quicker
            try:
                status = AllChem.UFFOptimizeMolecule(ligand,
                                                 maxIters=self.parameters.coordinate_generation.maximum_iterations)
            except:
                status = 1

            if status == 1:
                logger.debug(f"The 3D coordinate generation of molecule number {lig_obj.get_ligand_number()} " +
                             f"(smile: {lig_obj.get_smile()}) did not converge" +
                             f" - try increasing the number of maximum iterations.",)
                failed += 1
                if converged_only:
                    continue

            if self.parameters.protonate:  # Add hydrogen
                ligand = Chem.AddHs(ligand, addCoords=True)

            self.ligands[idx] = Ligand(smile=lig_obj.get_smile(),
                                       original_smile=lig_obj.get_original_smile(),
                                       ligand_number=lig_obj.get_ligand_number(),
                                       molecule=ligand)
            succeeded += 1

        if failed > 0:
            logger.debug(f"Of {len(self.ligands)}, {failed} could not be embedded.")
        logger.debug(f"In total, {succeeded} ligands were successfully embedded by RDkit.")

    def write_ligands(self):
        path = self.output.conformer_path
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)  # make path if not exist
        ligands_copy = [deepcopy(lig) for lig in self.ligands]

        if self.output.format.upper() == "SDF":
            writer = Chem.SDWriter(path)
            for lig in ligands_copy:
                if lig.get_molecule() is not None:
                    mol = deepcopy(lig.get_molecule())
                    mol.SetProp("_Name", str(lig.get_ligand_number()))
                    writer.write(mol)
            writer.close()
        else:
            raise Exception("Only SDF format supported.")
        logger.debug(f"Wrote {len(self.ligands)} molecules to file {path}.")
