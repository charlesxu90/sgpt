from copy import deepcopy


class Ligand:
    """This class bundles all information on a ligand, including all molecule instances present."""

    def __init__(self, smile: str, ligand_number: int, molecule=None, original_smile=None, name=None):
        # set attributes
        self._name = name
        self._smile = self._check_smile(smile)
        self._original_smile = original_smile
        self._ligand_number = self._check_ligand_number(ligand_number)
        self._molecule = molecule
        self._conformers = []

    def __repr__(self):
        return f"<Ligand id: {self.get_ligand_number()}, smile: {self.get_smile()}>"

    def __str__(self):
        return f"Ligand id: {self.get_ligand_number()}, smile: {self.get_smile()}, " \
               f"original_smile: {self.get_original_smile()}, " \
               f"has molecule: {True if self.get_molecule() is not None else False}."

    def get_clone(self):
        clone = Ligand(smile=self.get_smile(),
                       ligand_number=self.get_ligand_number(),
                       molecule=deepcopy(self.get_molecule()),
                       original_smile=self.get_original_smile())
        for conformer in self.get_conformers():
            clone.add_conformer(deepcopy(conformer))
        return clone

    def __copy__(self):
        return self.get_clone()

    def __deepcopy__(self, memo):
        return self.get_clone()

    def set_name(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def add_conformer(self, conformer):
        self._conformers.append(conformer)

    def set_conformers(self, conformers: list):
        self._conformers = conformers

    def get_conformers(self):
        return self._conformers

    def clear_conformers(self):
        self._conformers = []

    def set_molecule(self, molecule):
        self._molecule = molecule

    def get_molecule(self):
        return self._molecule

    def get_identifier(self):
        return str(self.get_ligand_number())

    @staticmethod
    def _check_smile(smile: str) -> str:
        if not isinstance(smile, str):
            raise ValueError(f"Field smile must be a string not of type {type(smile)}.")
        return smile

    def set_smile(self, smile: str):
        self._smile = self._check_smile(smile)

    def get_smile(self):
        return self._smile

    def set_original_smile(self, smile: str):
        self._original_smile = smile

    def get_original_smile(self):
        return self._original_smile

    @staticmethod
    def _check_ligand_number(ligand_number: int):
        if not isinstance(ligand_number, int) or ligand_number < 0:
            raise ValueError(f"Ligand number must be an integer value (minimally 0), not {ligand_number}.")
        return ligand_number

    def set_ligand_number(self, ligand_number: int):
        self._ligand_number = self._check_ligand_number(ligand_number)

    def get_ligand_number(self):
        return self._ligand_number

    @staticmethod
    def _add_title_to_molecule(molecule, title):
        molecule.SetProp("_Name", str(title))

    @staticmethod
    def _add_tag_to_molecule(molecule, tag, value):
        molecule.SetProp(tag, str(value))

    def add_tags_to_conformers(self):
        if len(self.get_conformers()) > 0:
            for conformer_number, conformer in enumerate(self.get_conformers()):
                self._add_title_to_molecule(conformer, self.get_identifier() + ':' + str(conformer_number))
                if self.get_name() is not None:
                    self._add_tag_to_molecule(conformer, "name", self.get_name())
                self._add_tag_to_molecule(conformer, "ligand_id", self.get_ligand_number())
                self._add_tag_to_molecule(conformer, "original_smiles", self.get_original_smile())
                self._add_tag_to_molecule(conformer, "smiles", self.get_smile())
