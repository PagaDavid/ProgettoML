import json
from subprocess import call


list_num_shift = [2, 10, 30]        # numero di istanti temporali da creare (es 2 -> | t-2 | t-1 | t |)

# leggo il file json
with open('config.json', 'r') as f:
    config = json.load(f)

for num_shift in list_num_shift:
    # scrivo sul file di configurazione un nuovo valore di num_shift
    config['num_shift'] = num_shift
    with open('config.json', 'w') as f:
        json.dump(config, f)

    all_stations = config['all_stations']  # prendo nelle x tutti i valori delle stazioni e utilizzo tante y quante sono le stazioni
    all_stations_single_y = config['all_stations_single_y']  # prendo nelle x tutti i valori delle stazioni e utilizzo una sola y di una stazione (ha effetto solo se all_stations = True)
    index_station_selected = config['index_station_selected']  # nel caso mi interessi solo una stazione, indico quale  (ha senso solo con una y)

    print "all_stations: " + str(all_stations)
    print "all_stations_single_y: " + str(all_stations_single_y)
    print "index_station_selected: " + str(index_station_selected)
    print "num_shift: " + str(num_shift)

    call(["python", "TimeSeriesComparison.py"])
    print "#"*50
