import pandas
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as matdates

# load data
dataset_with_missing_att = pandas.read_csv('dati/stations_3_LA_2000_2014_PRECIP_HLY_standard.csv', usecols=[0,5,6])
print dataset_with_missing_att

# elimino i campioni con valore di HPCP a 999.99 (significa missing attribute)
dataset = dataset_with_missing_att[dataset_with_missing_att.HPCP != 999.99]

#print dataset.loc[dataset.iloc[:,2].values == 999.99]
# sostituisco i valori 999.99 (dati mancanti) con 0
#dataset.replace(999.99, 0, inplace=True)

# plotto solo i dati temoprali (trasformando il dataframe in numpy array)
#plt.plot(dataset.iloc[:,2].values)
#plt.show()

# divido i dati tra le diverse stazioni
stations = set()
stations.update(dataset.STATION)
dataset_stations = []
for station in stations:
    dataset_stations.append(dataset[dataset.STATION == station])
#print dataset_stations

# considero solo UNA stazione: per ora la seconda perche' insieme alla terza sono molto vicine -> possibile estensione
# con i dati di entrambe le stazioni
first_station_dataset = dataset_stations[1]

#######################plotto solo i dati temoprali (trasformando il dataframe in numpy array)########################

# funzione che trasforma il formato di data attuale con uno maggiormente riconoscibile
def trasform_dataset_date_in_true_date(string_dates):
    dates = []
    for date in string_dates:
        new_date = date[0:4] + "-" + date[4:6] + "-" + date[6:]
        dates.append(new_date)
    return dates

date_string = first_station_dataset.iloc[:,1].values            # le date del dataset
new_format_date_string = trasform_dataset_date_in_true_date(date_string)        # trasformo il formato attuale di date in uno nuovo

# trasformo le stringhe in oggetti datetime
list_datetime = [datetime.strptime(data, '%Y-%m-%d %H:%M') for data in new_format_date_string]
list_matplot_dates = [matdates.date2num(d) for d in list_datetime]          # dai datetime ottengo date di matplotlib
plt.plot_date(list_matplot_dates, first_station_dataset.iloc[:,2].values, xdate=True)
plt.show()