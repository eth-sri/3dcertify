import transformations.manual_composition
import transformations.rotation
import transformations.shearing
import transformations.tapering
import transformations.twisting
from transformations.composition import Composition
from transformations.transformation import Transformation

_TRANSFORMATIONS = {
    'rotationx': transformations.rotation.RotationX,
    'rotationy': transformations.rotation.RotationY,
    'rotationz': transformations.rotation.RotationZ,
    'twistingz': transformations.twisting.TwistingZ,
    'shearingz': transformations.shearing.ShearingZ,
    'taperingz': transformations.tapering.TaperingZ,
    'rotationzx': transformations.manual_composition.RotationZX,
}


def parse_transformation(serialized: str) -> Transformation:
    split = serialized.strip().split('+')
    if len(split) < 1:
        raise ValueError("Invalid transformation, expecting at least one valid transformation.")
    elif len(split) == 1:
        transformation = _TRANSFORMATIONS[split[0].lower()]
        if transformation is None:
            raise ValueError(f"Unknown transformation {split[0]}, please specify a valid transformation or implement it.")
        return transformation()
    else:
        transformation = Composition(parse_transformation(split[0]), parse_transformation(split[1]))
        for i in range(2, len(split)):
            transformation = Composition(transformation, parse_transformation(split[i]))
        return transformation
