from typing import Dict, Union, Type
from preprocessing.preprocessor.interface import PreprocessorInterface
from preprocessing.preprocessor import sv1, nsv1, single_design_ns_v1


class ProcessorFactory:
    CONVERT_MODES: Dict[int, Dict[str, Union[str, Type[PreprocessorInterface], Dict[str, str]]]] = {
        1: {"name": "Aperture Stop v1", "class": sv1.PreprocessorSV1},
        2: {"name": "No Aperture Stop v1", "class": nsv1.PreprocessorNSV1},
        3: {"name": "GAGAGAGA(N=8) No Aperture Stop v1", "class": single_design_ns_v1.PreprocessorSingleDesignNSV1, "args": {"sequence": "GAGAGAGA"}},
        4: {"name": "GA(N=2) No Aperture Stop v1", "class": single_design_ns_v1.PreprocessorSingleDesignNSV1,
            "args": {"sequence": "GA"}},
        5: {"name": "GGA(N=3) No Aperture Stop v1", "class": single_design_ns_v1.PreprocessorSingleDesignNSV1,
            "args": {"sequence": "GGA"}},
        6: {"name": "GAGA(N=4) No Aperture Stop v1", "class": single_design_ns_v1.PreprocessorSingleDesignNSV1,
            "args": {"sequence": "GAGA"}},
    }

    @staticmethod
    def create(mode: int) -> PreprocessorInterface:
        mode_info = ProcessorFactory.CONVERT_MODES.get(mode)
        if mode_info and mode_info.get("class"):
            if args := mode_info.get("args"):
                return mode_info.get("class")(**args)
            else:
                return mode_info.get("class")()
        else:
            raise ValueError(f"Invalid Convert Mode Number: {mode}")

    @staticmethod
    def get_available_convert_mode() -> int:
        print("Available Convert Mode:")
        for k, v in ProcessorFactory.CONVERT_MODES.items():
            print(f"[{k}]: {v['name']}")

        while True:
            try:
                usr_input = int(input("Select Convert Mode Number: "))
                if usr_input in ProcessorFactory.CONVERT_MODES:
                    return usr_input
                else:
                    print("Invalid choice. Please select a valid Convert Mode Number.")
            except ValueError:
                print("Invalid input_object. Please enter a number.")

    @staticmethod
    def create_from_user_input() -> PreprocessorInterface:
        return ProcessorFactory.create(ProcessorFactory.get_available_convert_mode())


class ProcessorCommand:
    def __init__(self, processor):
        self.processor = processor

    def execute(self, df_to_conv):
        return self.processor.encode(df_to_conv)
