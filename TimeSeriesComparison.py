import pandas
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as matdates
import numpy
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
import json
from DNNTimeSeries import DNN
from LSTMTimeSeries import LSTMTS
from SVRegression import SVRTS
import numpy as np

# funzione che carica i dati e fa un minimo preprocessing
def load_data(filename):
    # carico dati
    dataset = pandas.read_csv(filename, usecols=[1, 2, 7])      # prendo solo la data, la stazione, hpcp

    # elimino i campioni con valore di HPCP a 999.99 (significa missing attribute)
    #dataset = dataset_with_missing_att[dataset_with_missing_att.HPCP != 999.99]
    return dataset

# funzione che plotta i dati creando un grafico x=tempo y=valori temporali
# parametri:
#     - date_string: lista temporali di date nel formato (2001-02-14 15:00:00)
#     - temporal_data: lista dei dati temporali (in questo caso hpcp)
def plot_data(date_string, temporal_data):
    # trasformo le stringhe in oggetti datetime
    list_datetime = [datetime.strptime(data, '%Y-%m-%d %H:%M:%S') for data in date_string]
    list_matplot_dates = [matdates.date2num(d) for d in list_datetime]  # dai datetime ottengo date di matplotlib
    plt.plot_date(list_matplot_dates, temporal_data, xdate=True)
    plt.show()

#funzione che trasforma il dataset in input con una rappresentazione adatta per il supervised learning per le time series
# parametri:
#     - dataset: matrice numpy con le sole informazioni temporali [[val1], [val2], ...]
#     - num_shift: quanti istanti passati considerare
def create_dataset_for_supervised_learning(dataset, num_shift = 1):
    dataset_for_sl = DataFrame(dataset)
    list_precip_hly_shifted = []                # lista di dataframe con sequenze temporali shiftate
    #for num_col in range(1, dataset.shape[1] + 1):
    for ns in range(1, num_shift + 1):
        precip_hly_shifted = dataset_for_sl.shift(ns)       # shifto la sequenza temporale avanti di un ns di step
        #print precip_hly_shifted
        list_precip_hly_shifted.insert(0, precip_hly_shifted)
    list_precip_hly_shifted.append(dataset_for_sl)  # appendo anche la sequnza originale
    dataset_for_sl = concat(list_precip_hly_shifted, axis=1)        # concateno le sequenze mantenendo una unica colonna di conteggio del dataframe
    return dataset_for_sl

# funzione che restituisce il numero totale di anni nel dataset e l'anno piu' recente
# parametri:
#    - dataset: dataframe con dateindex come indice
def find_last_years_and_num_years_of_dataset(dataset):
    list_years = dataset.index.year         # ottengo la lista di anni ripetuti
    list_distinct_years = set(list_years)
    return len(list_distinct_years) - 2, list_years[len(list_years) - 2]            # NOTA: faccio -2 per entrambi perche' viene fuori il 2014 che ha solo un'ora, quindi non lo considero

# funzione che crea un dataset di anni consecutivi di una stessa feature (metodo previous_years)
# presuppone che il dataset, oltre alle feature significative (in questo caso solo hpcp (ma che puo' essere di diverse stazioni)), contenga solamente
# una colonna di date predisposta come indice su cui fare query agevolmente
# parametri:
#     - dataset: dataframe pandas con feature di interesse e colonna di date
#     - num_shift: numero di anni da concatenare
def create_dataset_for_supervised_learning_for_previous_years(dataset, num_shift):
    max_num_years_dataset, end_date = find_last_years_and_num_years_of_dataset(dataset)
    list_df_hpcp_per_years = []

    # controllo che il numero di shift indicati non cerchi un numero di anni non presente nel dataset
    if (num_shift > max_num_years_dataset):
        num_shift = max_num_years_dataset

    # parto dall'ultimo anno e pian piano prendo tutti gli anni precedenti necessari
    for ns in range(1, num_shift + 2):          # faccio +2 perche' a differenza del metodo con temporal shift conto anche l'iterazione del valore da predire
        # prendo un anno
        lower_bound_date = datetime(end_date, 1, 1, 1)
        upper_bound_date = datetime(end_date, 12, 31, 23)
        end_date -= 1           # tolgo un anno

        df_hpcp_index_date = dataset.ix[lower_bound_date:upper_bound_date]  # faccio la query sull'indice delle date
        df_hpcp_index_date.reset_index(inplace=True)  # ristabilisco l'indice originario cancellando quello sulle date
        df_hpcp_index_date = df_hpcp_index_date.drop("DATE", axis=1)
        list_df_hpcp_per_years.insert(0, df_hpcp_index_date)
    dataset_for_sl = concat(list_df_hpcp_per_years, axis=1)
    return dataset_for_sl

# funzione che crea nel dataframe di input un'indicizzazione sul campo date
# parametri:
#    - dataframe: dataframe pandas con almeno una colonna di date su cui indicizzare il dataframe
def create_index_of_dates_in_dataframe(dataframe):
    # sovrascrivo le date attuali con un formato che puo' essere indicizzato
    dataframe = dataframe.copy()            # espediente per evitare di sollevare, nella prossima riga, il settingwithcopywarning
                                            # (in pratica creando una copia, si evita di modificare un slide dell'originale)
    dataframe.loc[:, 'DATE'] = pandas.DatetimeIndex(dataframe.DATE)
    dataframe.set_index(keys='DATE', inplace=True)
    return dataframe

# funzione che permette di plottare le predizioni dei modelli:
# parametri:
#     - list_true_temporal_data: dati temporali veri
#     - list_predict_temporal_data: dati temporali con le previsioni dei modelli
#     - index_subplot: indice del subplot
#     - type_model: stringa rappresentante il tipo di modello
def plot_predictions(list_true_temporal_data, list_predict_temporal_data, index_subplot, type_model):
    plt.subplot(3, 1, index_subplot)
    plt.plot(list_true_temporal_data)
    plt.plot(list_predict_temporal_data)
    plt.ylabel('Scaled HPCP')
    plt.xlabel('Samples')
    plt.legend(('True data', type_model + ' predictions'), loc='upper right')
    plt.title(type_model)
    if index_subplot == 2:          # se era l'ultimo grafico, visualizzo il plot di comparazione
        plt.draw()
        #plt.show(block=False)

# funzione utile per plottare la ffrequenza di pioggia rilevata dalle stazioni
# df: dataframe con indice di date e come altre colonne gli hpcp di ogni stazione
def hpcp_analysis(df):
    # costruisco una lista con valori numerici indicanti il numero di ore consecutive senza pioggia e date
    # indicanti ore di pioggia
    list_precip_and_absence_precip = []
    consecutives_hours_absence_precip = 0
    for index, hpcp in df.iterrows():
        if (hpcp.HPCP_0 > 0):           # se piove
            if (consecutives_hours_absence_precip != 0):        # per non appendere lo 0 nel caso di ore di pioggia consecutive
                list_precip_and_absence_precip.append(consecutives_hours_absence_precip)
            list_precip_and_absence_precip.append(index)
            consecutives_hours_absence_precip = 0
        else:                           # se non piove
            consecutives_hours_absence_precip += 1
    #print list_precip_and_absence_precip
    # costruisco una lista di valori numerici indicanti il numero di ore consecutive in cui piove (conteggio le date della lista precedente)
    list_consecutives_hours_precip = []
    num_consecutives_hours_precip = 0
    for val, count in zip(list_precip_and_absence_precip, range(len(list_precip_and_absence_precip))):
        if type(val) != int:            # se e' una data
            num_consecutives_hours_precip += 1
            if count == len(list_precip_and_absence_precip) -1:         # se termina la lista con ore consecutive di pioggia
                list_consecutives_hours_precip.append(num_consecutives_hours_precip)
        else:                           # se e' un numero
            if num_consecutives_hours_precip != 0:              # nel caso in cui non ci siano numeri consecutivi
                list_consecutives_hours_precip.append(num_consecutives_hours_precip)
                num_consecutives_hours_precip = 0
    #print list_consecutives_hours_precip

    # plotto la lista
    fig = plt.figure()
    plt.plot(list_consecutives_hours_precip)
    mean = np.mean(list_consecutives_hours_precip)
    plt.plot([mean for i in range(len(list_consecutives_hours_precip))])            # plotto la media
    plt.ylabel('Number of consecutive hours of rain')
    plt.xlabel('Number of slot of consecutive hours of rain')
    plt.title("Station 1")
    plt.show()
    fig.savefig("Hours_consecuitve_precip_station1.png")
    plt.close(fig)

    # plotto l'istogramma di frequenza dei valori di ore consecutive di pioggia
    fig = plt.figure()
    plt.hist(list_consecutives_hours_precip, bins=range(max(list_consecutives_hours_precip) + 2))
    plt.xlabel('Number of consecutive hours of rain')
    plt.ylabel('Frequencies of values')
    plt.title("Histogram consecutive hours of rain station 1")
    plt.show()
    fig.savefig("Histogram_hours_consecuitve_precip_station1.png")
    plt.close(fig)


if __name__ == '__main__':

    # carico i dati
    filename = 'dati/complete_stations_3_LA_2000_2014_PRECIP_HLY_standard.csv'
    dataset = load_data(filename)
    #print dataset

    # carico i parametri da un file di configurazione
    with open('config.json', 'r') as f:
        config = json.load(f)

    all_stations = config['all_stations']  # prendo nelle x tutti i valori delle stazioni e utilizzo tante y quante sono le stazioni
    all_stations_single_y = config['all_stations_single_y']  # prendo nelle x tutti i valori delle stazioni e utilizzo una sola y di una stazione (ha effetto solo se all_stations = True)
    index_station_selected = config['index_station_selected']  # nel caso mi interessi solo una stazione, indico quale  (ha senso solo con una y)
    num_shift = config['num_shift']
    temporal_shift = config['temporal_shift']
    num_stations = 1  # identificatore delle Y che mi interesssano
    num_stations_x = 1  # identificatore delle x che mi interesssano

    print "all_stations: " + str(all_stations)
    print "all_stations_single_y: " + str(all_stations_single_y)
    print "index_station_selected: " + str(index_station_selected)
    print "num_shift: " + str(num_shift)
    print "temporal_shift: " + str(temporal_shift)

    # trovo le stazioni a disposizione
    stations = set()
    stations.update(dataset.STATION)

    # controllo dei parametri presi in input
    if (index_station_selected < 0 or index_station_selected >= len(stations)):
        print "Scegliere un valore di stazione tra [0," + str(len(stations) - 1) + "]"
        exit(1)
    if (num_shift <= 0 or num_shift >= dataset.shape[0] / len(stations)):
        print "Scegliere un valore di shift tra [1," + str(dataset.shape[0] / len(stations) - 1) + "]"
        exit(1)

    # divido i dati fra le varie stazioni
    dataset_stations = []
    for station in stations:
        dataset_stations.append(dataset[dataset.STATION == station])
    all_date_in_common_for_all_stations = list(dataset_stations[0].DATE)        # indice 0, ma andava bene qualsiasi indice di stazione

    if not all_stations:
        # considero solo UNA stazione: per ora la seconda perche' insieme alla terza sono molto vicine -> possibile estensione
        # con i dati di entrambe le stazioni
        print "ONE STATION"
        first_station_dataset = dataset_stations[index_station_selected]

        #####################inizio plot data ###################
        # cerco gli indici delle colonne DATE e HPCP
        df_columns = list(first_station_dataset.columns.values)
        index_date = df_columns.index('DATE')
        index_hpcp = df_columns.index('HPCP')

        # plotto i dati
        # plot_data(first_station_dataset.iloc[:, index_date].values, first_station_dataset.iloc[:, index_hpcp].values)
        #####################fine plot data ###################

        if (not temporal_shift):                # metodologia di combinazione dei dati della stazione di tipo previous_years

            # manipolo il dataframe (mettendo come indice la data) in modo da poter effettuare query su campioni di particolari date in maniera piu' diretta
            first_station_dataset = create_index_of_dates_in_dataframe(first_station_dataset)
            first_station_dataset = first_station_dataset.drop("STATION", axis=1)
            dataset_for_sl = create_dataset_for_supervised_learning_for_previous_years(first_station_dataset, num_shift)
            first_station_dataset.reset_index(inplace=True)  # ristabilisco l'indice originario cancellando quello sulle date

        else:                       # metodologia di combinazione dei dati della stazione di tipo temporal_shift

            first_station_dataset = first_station_dataset.drop(["STATION", "DATE"], axis=1).values
            dataset_for_sl = create_dataset_for_supervised_learning(first_station_dataset, num_shift)

    else:           # considero TUTTE le stazioni

        num_stations = len(stations)            # aggiorno la variabile di stazioni che considero (utile per l'individuazione delle Y)
        num_stations_x = len(stations)

        # estraggo tutti i valori di hpcp di ogni stazione (l'ordine e' gia' corretto) concatenandoli, poi cro un unico dataframe che li contiene (hpcp stazione 1 | hpcp stazione 2 | hpcp stazione 3)
        df_all_stations = pandas.DataFrame()
        list_df_single_station = []
        index_station = 0
        for df_single_station in dataset_stations:
            df_single_station = df_single_station.drop(["STATION", "DATE"], axis=1)
            df_app = pandas.DataFrame()
            df_app['HPCP_' + str(index_station)] = df_single_station.HPCP.values
            list_df_single_station.append(df_app)
            index_station += 1
        df_all_stations = concat(list_df_single_station, axis=1)

        if (not temporal_shift):                   # metodologia di combinazione dei dati della stazione di tipo previous_years
            print "previous_years"

            # aggiungo le date per creare un dataframe indicizzato su di esse
            df_dates = dataset_stations[0].DATE
            list_df = []
            list_df.append(df_all_stations)
            list_df.append(df_dates)
            df_all_stations = concat(list_df, axis=1)

            df_all_stations = create_index_of_dates_in_dataframe(df_all_stations)
            #hpcp_analysis(df_all_stations)
            dataset_for_sl = create_dataset_for_supervised_learning_for_previous_years(df_all_stations, num_shift)
            df_all_stations.reset_index(inplace=True)  # ristabilisco l'indice originario cancellando quello sulle date

        else:                                   # metodologia di combinazione dei dati della stazione di tipo temporal_shift

            # trasformo il dataset attuale in un dataset per il learning supervisionato
            dataset_for_sl = create_dataset_for_supervised_learning(df_all_stations.values, num_shift)

        print "all_station"

        # se sono nel caso in cui mi interessano tutte le x delle stazioni, ma voglio una sola y
        if (all_stations_single_y):
            indexes_columns_x = num_shift*num_stations      # trovo l'indice dell'ultima colonna delle x
            # prendo solo la y della stazione di interesse
            list_parts_general_df = []
            list_parts_general_df.append(dataset_for_sl.iloc[:,:indexes_columns_x])
            list_parts_general_df.append(dataset_for_sl.iloc[:,indexes_columns_x+index_station_selected])
            dataset_for_sl = concat(list_parts_general_df, axis=1)

            # anche se sono nella parte di codice di estrazione delle x per tutte le stazioni, ora mi devo ricondurre al caso
            # di una sola y, ovvero al caso opposto a questo
            num_stations = 1
            all_stations = False

            print "all_station_x_one_station_y"

    dataset_for_sl = dataset_for_sl.dropna()          # CANCELLO le righe contenenti i NAN introdotti con lo shift

    # scalo eventualmente (dato che il massimo e' 1.29 e il minimo e' 0) i dati
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_for_sl = scaler.fit_transform(dataset_for_sl)

    # creo la suddivisione tra training set, development set e test set (60% per il training e 20% per il development e 20% per il test)
    train_size = int(round(len(dataset_for_sl) * 0.60))
    dev_size = int(round(len(dataset_for_sl) * 0.20))
    test_size = int(len(dataset_for_sl) - (train_size + dev_size))
    training_set = dataset_for_sl[:train_size,:]
    development_set = dataset_for_sl[train_size:train_size+dev_size,:]
    test_set = dataset_for_sl[len(dataset_for_sl) - test_size:,:]

    # divido sia per il training set, che per il dev_test, che per il test set, le variabili X con le Y
    x_training_set = training_set[:, :-num_stations]
    y_training_set = training_set[:, -num_stations:]
    x_dev_set = development_set[:, :-num_stations]
    y_dev_set = development_set[:, -num_stations:]
    x_test_set = test_set[:, :-num_stations]
    y_test_set = test_set[:, -num_stations:]

    # nel caso di singola y, per evitare il triggering della DATACONVERSION WARNING relativa ad una y fatta cosi' [[val 1], ... , [val n]],
    # la trasformo in [val 1, ..., val n]
    if (not all_stations or all_stations_single_y):
        y_training_set = y_training_set.flatten()
        y_dev_set = y_dev_set.flatten()
        y_test_set = y_test_set.flatten()
    # creo una rappresentazione a lista delle y vere per facilitarne il plot
    list_true_temporal_data = list(y_training_set) + list(y_dev_set) + list(y_test_set)  # concateno tutti i valori di y

    ######################## alleno i tre modelli a confronto #################################
    # creo un parametro aggiuntivo ai tre modelli per sapere se devo effettuare il training (train = True) e la stima dei parametri (train = False)
    train = True

    fig = plt.figure(figsize=(16, 14))

    ############# svr
    print "SVR:"
    svr = SVRTS(x_training_set, y_training_set, x_dev_set, y_dev_set, x_test_set, y_test_set, action=train, output_multi=all_stations)
    if (train):
        #list_predict_temporal_data = list(y_training_set) + list(y_dev_set) + [pred for pred in svr.predictions_test]
        #plot_predictions(list_true_temporal_data, list_predict_temporal_data, 1, 'SVR')
        list_predict_temporal_data = [pred for pred in svr.predictions_test]
        plot_predictions(list(y_test_set), list_predict_temporal_data, 1, 'SVR')

    ########### dnn
    print "#" * 50
    print "DNN:"
    dnn = DNN(x_training_set, y_training_set, x_dev_set, y_dev_set, x_test_set, y_test_set, x_training_set.shape[1], action=train, output_multi=all_stations)
    if (train):
        #list_predict_temporal_data = list(y_training_set) + list(y_dev_set) + [pred for pred in dnn.predictions_test]
        #plot_predictions(list_true_temporal_data, list_predict_temporal_data, 2, 'DNN')
        list_predict_temporal_data = [pred for pred in dnn.predictions_test]
        plot_predictions(list(y_test_set), list_predict_temporal_data, 2, 'DNN')

    ############# lstm
    print "#" * 50
    print "LSTM:"
    #faccio il reshape dei dati (num_samples, time_steps, num_features)
    x_training_set_lstm = numpy.reshape(x_training_set, (x_training_set.shape[0], num_shift, num_stations_x))
    x_dev_set_lstm = numpy.reshape(x_dev_set, (x_dev_set.shape[0], num_shift, num_stations_x))
    x_test_set_lstm = numpy.reshape(x_test_set, (x_test_set.shape[0], num_shift, num_stations_x))
    lstm = LSTMTS(x_training_set_lstm, y_training_set, x_dev_set_lstm, y_dev_set, x_test_set_lstm, y_test_set, num_shift,
                  action=train, output_multi=all_stations)
    if (train):
        #lstm_predictions = [pred for pred in lstm.predictions_test]
        #list_predict_temporal_data = list(y_training_set) + list(y_dev_set) + lstm_predictions
        #plot_predictions(list_true_temporal_data, list_predict_temporal_data, 3, 'LSTM')
        lstm_predictions = [pred for pred in lstm.predictions_test]
        list_predict_temporal_data = lstm_predictions
        plot_predictions(list(y_test_set), list_predict_temporal_data, 3, 'LSTM')

    if (train):
        fig.savefig("Comparison_models_predictions.png")
        plt.close(fig)