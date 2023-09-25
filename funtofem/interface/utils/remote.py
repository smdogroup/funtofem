__all__ = ["Remote"]
import os

class Remote:
    def __init__(
        self,
        analysis_file,
        main_dir,
        output_name="f2f_analysis",
        nprocs=1,
        aero_name="fun3d",
        struct_name="tacs",
    ):
        """

        Manages remote analysis calls for a FUN3D / FUNtoFEM driver call

        Parameters
        ----------
        nprocs: int
            number of procs for the system call to the Fun3dOnewayAnalyzer
        analyzer_file: os filepath
            the location of the subprocess file for the Fun3dOnewayAnalyzer (my_fun3d_analyzer.py)
        main_dir: filepath
            location of the fun3d directory for meshes if FUN3D, one level above the scenario folders
        output_file: filepath
            optional location to write an output file for the forward and adjoint analysis
        """
        self.analysis_file = analysis_file
        self.main_dir = main_dir
        self.nprocs = nprocs
        self.output_name = output_name
        self.aero_name = aero_name
        self.struct_name = struct_name

    @classmethod
    def paths(cls, main_dir, aero_name="fun3d", struct_name="struct"):
        return cls(
            analysis_file=None,
            main_dir=main_dir,
            aero_name=aero_name,
            struct_name=struct_name,
        )

    @classmethod
    def fun3d_path(cls, main_dir, filename):
        return os.path.join(main_dir, filename)

    @property
    def struct_sens_file(self):
        return os.path.join(self.main_dir, f"{self.struct_name}.sens")

    @property
    def aero_sens_file(self):
        return os.path.join(self.main_dir, f"{self.aero_name}.sens")

    @property
    def output_file(self):
        return os.path.join(self.main_dir, f"{self.output_name}.txt")

    @property
    def bdf_file(self):
        return os.path.join(self.main_dir, f"{self.struct_name}.bdf")

    @property
    def dat_file(self):
        return os.path.join(self.main_dir, f"{self.struct_name}.dat")

    @property
    def design_file(self):
        return os.path.join(self.main_dir, "funtofem.in")

    @property
    def functions_file(self):
        return os.path.join(self.main_dir, "funtofem.out")