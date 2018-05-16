import pandas
from pandas import concat

# funzione che trasforma il formato di data attuale (string_dates 20010214 15:00:00) con uno maggiormente riconoscibile (2001-02-14 15:00:00)
def trasform_dataset_date_in_true_date(string_dates):
    dates = []
    for date in string_dates:
        new_date = date[0:4] + "-" + date[4:6] + "-" + date[6:] + ":00"
        dates.append(new_date)
    return dates

# funzione che inserisce in un dizionario i valori costanti di ogni stazione
def getDataStation(station, data_station, info_stations):
    info_stations[station] = {}
    info_stations[station]['ELEVATION'] = data_station.ELEVATION.values[0]
    info_stations[station]['LATITUDE'] = data_station.LATITUDE.values[0]
    info_stations[station]['LONGITUDE'] = data_station.LONGITUDE.values[0]
    info_stations[station]['STATION_NAME'] = data_station.STATION_NAME.values[0]

if __name__ == '__main__':

    # creo un dataframe con tutti gli istanti temporali richiesti
    dataframe_dataset = pandas.DataFrame()
    # con to_pydatetime() trasformo le dateindex di pandas in datetime di python
    dataframe_dataset['DATE'] = pandas.date_range(start = '2000-01-01 01:00', end = '2014-01-01', freq = '1H').to_pydatetime()

    # carico dati presi da internet
    filename = 'dati/stations_3_LA_2000_2014_PRECIP_HLY_standard.csv'
    dataset_with_missing_att = pandas.read_csv(filename, usecols=[0,1,2,3,4,5,6])           # trascuro le colonne relative ai flag

    # modifico le  date ottenute con un formato migliore
    date_string = dataset_with_missing_att.DATE
    new_format_date_string = trasform_dataset_date_in_true_date(date_string)
    # creo un dataframe con tali date di tipo datetime per modificare il dataframe dei dati originari
    df_app = pandas.DataFrame()
    df_app['DATE'] = pandas.DatetimeIndex(new_format_date_string).to_pydatetime()
    # sostituisco la colonna DATE con i valori giusti
    dataset_with_missing_att = dataset_with_missing_att.drop('DATE', axis = 1)
    dataset_with_missing_att['DATE'] = df_app.DATE.values

    #elimino i campioni con valore di HPCP a 999.99 (significa missing attribute)
    dataset = dataset_with_missing_att[dataset_with_missing_att.HPCP != 999.99]

    # divido i dati tra le diverse stazioni
    stations = set()
    stations.update(dataset.STATION)
    info_stations = {}      # creo un dizionario che conterra' le informazioni costanti di ogni stazione
    list_df_stations = []       # lista di dataframe di ogni stazione
    for station in stations:
        data_station = dataset[dataset.STATION == station]
        getDataStation(station, data_station, info_stations)

        # faccio il merge dei due dataframe (quello dei dati originali con date corrette e quello con tutte le ore richieste)
        df_station = pandas.merge_asof(dataframe_dataset, data_station, by = 'DATE', on = 'DATE')
        # il datagramma ottenuto presenta i valori del file scaricato da internet solamente in occorrenza di un match di una data con l'altro dataframe,
        # gli altri valori (di qualsiasi colonna) sono messi a NaN, quindi per mettere i valori mancanti sfrutto il dizionario preparato precedentemente
        df_station['STATION'] = station
        df_station['ELEVATION'] = info_stations[station]['ELEVATION']
        df_station['LATITUDE'] = info_stations[station]['LATITUDE']
        df_station['LONGITUDE'] = info_stations[station]['LONGITUDE']
        df_station['STATION_NAME'] = info_stations[station]['STATION_NAME']

        # adesso sostituisco i NaN (della precipitazione HPCP) con degli 0
        df_station = df_station.fillna(0.0)
        list_df_stations.append(df_station)

    # creo un unico dataframe e lo stampo
    final_dataset = pandas.DataFrame()
    final_dataset = concat(list_df_stations, axis=0)
    final_dataset.to_csv("dati/complete_stations_3_LA_2000_2014_PRECIP_HLY_standard.csv")