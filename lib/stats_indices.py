import numpy as np
import numexpr as ne

def calc_indices(index, B02=None, B03=None, B04=None, B05=None, B06=None, B07=None, B08=None, B8A=None, B11=None, B12=None):
    if index == 'chlre':  # Red Edge Chlorophyll Index
        return ne.evaluate("B05 / B08")
    elif index == 'rendvi':  # Red Edge NDVI
        return ne.evaluate('(B08 - B06) / (B08 + B06)')
    elif index == 's2rep':  # Sentinel 2 Red Edge Position
        return ne.evaluate('705 + 35 * ((((B07 + B04) / 2) - B05) / (B06 - B05))')
    elif index == 'ireci':  # Red Edge Chlorophyl Index
        return ne.evaluate('((B07 - B04) * B06) / B05')
    elif index == 'mcari':  # Modified Chlorophyll Absorption in Reflectance Index
        return ne.evaluate('((B05 - B04) - 0.2 * (B05 - B03)) * (B05 / B04)')
    elif index == 'arvi':  # Atmospherically Resistant Vegetation Index
        return ne.evaluate('(B08 - b) / (B08 + b)', {"b": ne.evaluate('2 * B04 - B02')}, {"B08": B08})
    elif index == 'savi':  # Soil adjusted vegetation index
        return ne.evaluate('((B08 - B04) / ((B08 + B04) + 0.428)) * 1.856')
    elif index == 'msavi2':  # Modified soil adjusted vegetation index v2
        return ne.evaluate('(2 * B08 + 1 - sqrt((2 * B08 + 1) ** 2) - 8 * (B08 - B04)) / 2')
    elif index == 'gndvi':  # Green normalised difference vegetation index
        return ne.evaluate('(B08 - B03) / (B08 + B03)')
    elif index == 'ndvi':  # Normalised difference vegetation index
        return ne.evaluate('(B08 - B04) / (B08 + B04)')
    elif index == 'moist':  # Soil moisture index
        return ne.evaluate('(B8A - B11) / (B8A + B11)')
    elif index == 'ndwi':  # Normalised difference water index
        return ne.evaluate('(B08 - B11) / (B08 + B11)')
    elif index == 'ndwi2':  # Normalised difference water index v2
        return ne.evaluate('(B03 - B08) / (B03 + B08)')
    elif index == 'nbr':  # Normalised difference burn ratio
        return ne.evaluate('(B08 - B12) / (B08 + B12)')
    elif index == 'nvei':  # Non-elimination vegetation index
        return ne.evaluate('(B02 - B04) / (B08 + B04)')
    elif index == 'nbai':  # Built-up area index
        return ne.evaluate('(B12 - d) / (B12 + d)', {"d": ne.evaluate('B08 / B02')}, {"B12": B12, "B08": B08, "B02": B02})
    elif index == 'brba':  # Band ratio for built-up areas
        return ne.evaluate('(B03 / B08)')
    elif index == 'ndbi':  # Normalised difference built-up index
        return ne.evaluate('(B11 - B08) / (B11 + B08)')
    elif index == 'blfei':  # Built-up features extraction
        return ne.evaluate('(b - B11) / (b + B11)', {"b": ne.evaluate('(B03 + B04 + B12) / 3')})
    elif index == 'ibi':  # Built-up features extraction
        return ne.evaluate('(ndbi - ((savi + ndwi2) / 2)) / (ndbi + ((savi + ndwi2) / 2))', {
            "savi": ne.evaluate('((B08 - B04) / ((B08 + B04) + 0.428)) * 1.856'),
            "ndwi2": ne.evaluate('(B03 - B08) / (B03 + B08)'),
            "ndbi": ne.evaluate('(B11 - B08) / (B11 + B08)'),
        })

