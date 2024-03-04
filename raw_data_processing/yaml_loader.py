import yaml
from pydantic import BaseModel
from typing import List
import os


class LensDesign(BaseModel):
    epd: float  # entrance pupil diameter
    hfov: float  # half field of view
    hfov_all: List[float]  # half field of view parameters for ray tracing
    wave: List[float]   # designed wavelengths
    c: List[float]  # curvature of surface excluding the stop
    as_c: float  # curvature of the aperture stop, if None, this is -1
    c_all: List[float]  # curvature of surface including the stop
    d_all: List[float]  # diameter of surface including stop
    t_all: List[float]  # thickness including the stop
    nd: List[float]  # refractive index 屈折率
    v: List[float]  # abbe number
    efl: float  # effective focal length
    sequence: str  # GGA, GAGA, etc. used for ray tracing
    stop_idx: int  # index of the aperture stop
    # t: List[float]  # thickness without the stop
    # as_t: float  # thickness of the aperture stop, if None, this is -1
    independent_as: bool  # whether the aperture stop is independent or not
    file_name: str

    @staticmethod
    def from_yaml(file_path: str) -> "LensDesign":
        with open(file_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        data["sequence"] = data["sequence"][0]
        data["stop_idx"] = int(data["stop_idx"][0])
        data["epd"] = data["epd"][0]
        data["hfov_all"] = data["hfov"] #[abs(x) for x in data["hfov"]]
        data["hfov"] = abs(data["hfov"][-1])
        data["c_all"] = data["c"]
        data["d_all"] = data["d"]
        data["t_all"] = data["t"]
        data["efl"] = data["efl"][0]
        ld = LensDesign(
            **data, file_name=os.path.basename(file_path), as_c=-1.0, independent_as=False
        )

        i = ld.stop_idx - 1  # 0-indexed index of the aperture stop
        if ld.sequence[i] == "A":
            # move the information of the aperture stop
            ld.as_c = ld.c.pop(i)
            ld.independent_as = True
        return ld



if __name__ == "__main__":
    LD = LensDesign.from_yaml('C:/Users/KU/Desktop/AI/design/Lensview/yaml_jp/15576.ZMX.yml') 