"""### Convert geometry composition. ###

Convert geometries from multiparts and singleparts and vice versa.
This module is maintained for backward compatibility.
New code should import from buteo.core_vector.conversion.* submodules directly.
"""

# Re-export functions for backward compatibility
from buteo.core_vector.conversion.multipart import (
    check_vector_is_multipart,
    vector_multipart_to_singlepart,
    vector_singlepart_to_multipart,
)

from buteo.core_vector.conversion.multitype import (
    _convert_multitype,
    vector_change_multitype,
)

from buteo.core_vector.conversion.dimensionality import (
    _set_z_value_to_geometry,
    _set_m_value_to_geometry,
    vector_change_dimensionality,
)

from buteo.core_vector.conversion.combined import (
    vector_convert_geometry,
)

__all__ = [
    "check_vector_is_multipart",
    "vector_multipart_to_singlepart",
    "vector_singlepart_to_multipart",
    "_convert_multitype",
    "vector_change_multitype",
    "_set_z_value_to_geometry",
    "_set_m_value_to_geometry",
    "vector_change_dimensionality",
    "vector_convert_geometry",
]
