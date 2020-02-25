import pandas
from geopy.geocoders import Nominatim
nom = Nominatim(user_agent="unhcr_test")

csv_file_in = '../geometry/villages.csv'
csv_file_out = '../geometry/villages_geocoded.csv'

villages = pandas.read_csv(csv_file_in)

total = 0
success = 0

for index, row in villages.iterrows():
  try:
    location = nom.geocode(row['name'], country_codes='sy')
    villages.at[index, 'lat'] = location.latitude
    villages.at[index, 'lng'] = location.longitude
    success += 1
  except:
    pass

  total += 1

import pdb; pdb.set_trace();
villages.to_csv(csv_file_out, index_label='index')

print(f'Completed {success}/{total} ({((success / total) * 100):.2f}%) locations.')
