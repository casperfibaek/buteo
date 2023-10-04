import os
import string
import overpass
import pandas as pd
from tqdm import tqdm
import json

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/"

# concat(to_string(round("bottom", 7)), concat(',', to_string(round("left", 7))), concat(',', to_string(round("top", 7))), concat(',', to_string(round("right", 7))))
valid_tags = [
    "name",
    "amenity",
    "aeroway",
    "building",
    "craft",
    "emergency",
    "healthcare",
    "historic",
    "landuse",
    "leisure",
    "military",
    "place",
    "power",
    "public_transport",
    "railway",
    "office",
    "shop",
    "tourism",
]

features = [
    "amenity=fast_food",
    "amenity=food_court",
    "amenity=cafe",
    "amenity=restaurant",
    "amenity=pub",
    "amenity=bar",
    "amenity=bank",
    "amenity=clinic",
    "amenity=doctors",
    "amenity=hospital",
    "amenity=pharmacy",
    "amenity=dentist",
    "amenity=veterinary",
    "amenity=arts_centre",
    "amenity=cinema",
    "amenity=casino",
    "amenity=community_centre",
    "amenity=conference_centre",
    "amenity=events_venue",
    "amenity=fuel",
    "amenity=exhibition_centre",
    "amenity=planetarium",
    "amenity=theatre",
    "amenity=nightclub",
    "amenity=courthouse",
    "amenity=fire_station",
    "amenity=police",
    "amenity=post_office",
    "amenity=prison",
    "amenity=townhall",
    "amenity=crematorium",
    "amenity=funeral_hall",
    "amenity=internet_cafe",
    "amenity=marketplace",
    "amenity=place_of_mourning",
    "amenity=place_of_worship",
    "aeroway=terminal",
    "aeroway=aerodrome",
    "building=commercial",
    "building=industrial",
    "building=office",
    "building=retail",
    "building=warehouse",
    "building=church",
    "building=cathedral",
    "building=chapel",
    "building=mosque",
    "building=temple",
    "building=synagogue",
    "building=shrine",
    "building=supermarket",
    "building=fire_station",
    "building=police",
    "building=prison",
    "building=hospital",
    "building=museum",
    "building=military",
    "craft=agricultural_engines",
    "craft=atelier",
    "craft=bakery",
    "craft=blacksmith",
    "craft=boatbuilder",
    "craft=brewery",
    "craft=cabinet_maker",
    "craft=carpenter",
    "craft=electronics_repair",
    "craft=distillery",
    "craft=oil_mill",
    "emergency=ambulance_station",
    "healthcare=alternative",
    "healthcare=audiologist",
    "healthcare=birthing_center",
    "healthcare=chiropractor",
    "healthcare=dentist",
    "healthcare=midwife",
    "healthcare=occupational_therapist",
    "healthcare=optometrist",
    "healthcare=physiotherapist",
    "healthcare=psychologist",
    "healthcare=speech_therapist",
    "healthcare=blood_bank",
    "healthcare=blood_donation",
    "healthcare=vaccination_centre",
    "historic=church",
    "historic=cathedral",
    "historic=castle",
    "historic=mosque",
    "historic=tower",
    "landuse=commercial",
    "landuse=retail",
    "landuse=industrial",
    "landuse=warehouse",
    "landuse=cemetery",
    "landuse=religious",
    "leisure=adult_gaming_centre",
    "leisure=amusement_arcade",
    "military=barracks",
    "military=base",
    "military=office",
    "place=farm",
    "place=allotments",
    "power=transformer",
    "public_transport=station",
    "railway=station",
    "office=government",
    "shop=alcohol",
    "shop=bakery",
    "shop=beverages",
    "shop=brewing_supplies",
    "shop=butcher",
    "shop=cheese",
    "shop=chocolate",
    "shop=coffee",
    "shop=confectionery",
    "shop=convenience",
    "shop=farm",
    "shop=food",
    "shop=general",
    "shop=department_store",
    "shop=kiosk",
    "shop=mall",
    "shop=supermarket",
    "shop=wholesale",
    "shop=beauty",
    "shop=fabric",
    "shop=fashion",
    "shop=doityourself",
    "shop=electronics",
    "shop=garden_centre",
    "shop=yes",
    "tourism=guest_house",
    "tourism=hostel",
    "tourism=hotel",
    "tourism=motel",
    "tourism=museum",
    "tourism=chalet",
    "tourism=apartment",
    "tourism=zoo",
]

exluded_words = [
    "Escola",
    "Universidade",
    "Faculdade",
    "Colégio",
    "Curso",
    "Instrução",
    "Pedagogia",
    "Licenciatura",
    "Pedagógico",
    "Educação",
    "Ensino",
    "Aula",
    "Sala de aula",
    "Escuela",
    "Universidad",
    "Colegio",
    "Instituto",
    "Academia",
    "Facultad",
    "Preparatoria",
    "Educación",
    "Cursillo",
    "Pedagogía",
    "Sala de clase",
    "Estudio",
    "School",
    "University",
    "College",
    "Institute",
    "Academy",
    "Education",
    "Learning",
    "Course",
    "Tutorial",
    "Classroom",
    "Lecture",
    "Seminar",
    "Elimu",
    "Darasa",
    "Ilimi",
    "Klass",
    "Lakco",
    "Ufundo",
    "Iklasi",
    "Isifundo",
    "ⵜⴰⵣⴷⴰⵡⵉⵜ",
    "Tazdawit",
    "ⵜⴰⵍⵎⵉⵢⵜ",
    "Talmiyt",
    "ⴰⵙⵙⴼⴰⵔ",
    "Assfar",
    "Somo",
    "ትምህርት",
    "Timihirt",
    "ክፍል",
    "Kifil",
    "Studio",
    "مدرسة",
    "Madrasa",
    "جامعة",
    "Jami'a",
    "أكاديمية",
    "Akadimiya",
    "كلية",
    "Kulliya",
    "معهد",
    "Ma'had",
    "Shule",
    "Chuo Kikuu",
    "Chuo",
    "Akademia",
    "Makaranta",
    "Jami'a",
    "Koleji",
    "Ile-iwe",
    "Ile-iwe giga",
    "Akademi",
    "Isikole",
    "Yunivesithi",
    "Akhawunti",
    "ትምህርት ቤት",
    "Timihirt Bet",
    "ዩኒቨርሲቲ",
    "Yuniversiti",
    "Tasdawit",
    "Tasdawit Urraz",
    "Iskuul",
    "Jaamacad",
    "Skool",
    "Universiteit",
    "Akademie",
    "Kollege",
    "École",
    "Université",
    "Collège",
    "Institut",
    "Académie",
    "Éducation",
    "Apprentissage",
    "Cours",
    "Tutoriel",
    "Classe",
    "Conférence",
    "Séminaire",
    "Studio"
]
excluded_words = [w.lower() for w in exluded_words]

bboxes = pd.read_csv(os.path.join(FOLDER, "south-america_grid.csv"))
api = overpass.API(timeout=16000)

for idx, row in tqdm(enumerate(bboxes["bbox"].items()), total=len(bboxes)):
    bbox_float = tuple([float(item) for item in row[1].split(',')])
    south, west, north, east = bbox_float
    bbox = str(bbox_float)

    nodes = ''.join([f'node["{feature.split("=")[0]}"="{feature.split("=")[1]}"]{bbox};' for feature in features])
    nodes = nodes.translate({ord(c): None for c in string.whitespace})
    relations = nodes.replace("node", "relation")
    ways = nodes.replace("node", "way")

    query = nodes + relations + ways

    qq = f"[out:json]; ({query}); out center;"

    response = api.get(qq, build=False, responseformat="json")

    if len(response["elements"]) == 0:
        print(f"No features found for bbox {bbox} and idx {idx}")
        continue

    elements = []
    for row in response["elements"]:
        row_dict = {}
        skip = False
        try:
            row_dict["type"] = row["type"]
            row_dict["id"] = row["id"]

            if "center" in row:
                row_dict["lat"] = row["center"]["lat"]
                row_dict["lon"] = row["center"]["lon"]
            else:
                row_dict["lat"] = row["lat"]
                row_dict["lon"] = row["lon"]

            for tag in row["tags"]:
                if tag in ["id", "lat", "lon", "type"]:
                    continue
                elif tag not in valid_tags:
                    continue
                elif isinstance(tag, str) and tag.lower() in excluded_words:
                    skip = True
                    continue 
                elif tag not in row_dict:
                    row_dict[tag] = str(row["tags"][tag])
                else:
                    row_dict[tag] = str(row_dict[tag]) + ";" + str(row["tags"][tag])
        except:
            import pdb; pdb.set_trace()

        if not skip:
            elements.append(row_dict)

    response_df = pd.read_json(json.dumps(elements), orient="records")
    out_csv = os.path.join(FOLDER, f"south-america_grid_{idx}.csv")
    out_gpkg = os.path.join(FOLDER, f"south-america_grid_{idx}.gpkg")
    response_df.to_csv(out_csv, index=False)

    # call command line
    ogr_options = '-oo X_POSSIBLE_NAMES=lon -oo Y_POSSIBLE_NAMES=lat -a_srs "EPSG:4326"'
    ogr_call = f'ogr2ogr "{out_gpkg}" "{out_csv}" {ogr_options}'
    os.system(ogr_call)

    # remove csv
    if os.path.exists(out_csv):
        os.remove(out_csv)
