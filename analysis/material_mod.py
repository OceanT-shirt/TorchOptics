
from rayopt.material import Material, AbbeMaterial, ModelMaterial, CoefficientsMaterial

class Material_Mod(Material):
    @classmethod
    def make(cls, name):
        if name is None:
            return None
        if isinstance(name, Material):
            return name
        if isinstance(name, dict):
            return super(Material, self).make(name)
        if type(name) is float:
            return ModelMaterial(n=name)
        if type(name) is tuple:
            return AbbeMaterial(n=name[0], v=name[1])
        try:
            return AbbeMaterial.from_string(name)
        except ValueError:
            pass
        parts = name.lower().split("/")
        name = parts.pop()
        source, catalog = None, None
        if parts:
            catalog = parts.pop()
        if parts:
            source = parts.pop()
        if catalog in (None, "basic") and name in basic:
            return basic[name]
        from rayopt.library import Library
        lib = Library.one()
        return lib.get("material", name, catalog, source)


# http://refractiveindex.info
vacuum = ModelMaterial(name="vacuum", catalog="basic", solid=False)
mirror = Material(name="mirror", catalog="basic", solid=False, mirror=True)
air = CoefficientsMaterial(
    name="air", catalog="basic", typ="gas", solid=False,
    coefficients=[.05792105, .00167917, 238.0185, 57.362])
basic = dict((m.name, m) for m in (vacuum, air, mirror))