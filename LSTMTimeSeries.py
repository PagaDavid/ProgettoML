import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras import backend as K
from sklearn.metrics import r2_score
from keras.optimizers import Adam

class LSTMTS:

	def __init__(self, x_training_set, y_training_set, x_dev_set, y_dev_set, x_test_set, y_test_set, time_steps, action, output_multi):
		self.x_training_set = x_training_set
		self.y_training_set = y_training_set
		self.x_dev_set = x_dev_set
		self.y_dev_set = y_dev_set
		self.x_test_set = x_test_set
		self.y_test_set = y_test_set
		self.time_steps = time_steps
		self.action = action  		# True per il training, false per la stima dei parametri
		self.output_multi = output_multi		# True se output multiplo, altrimenti False

		# inizializzo parametri del modello
		self.model = None
		self.predictions_training = None
		self.predictions_test = None
		self.score_training = None
		self.score_test = None
		self.accuracy_tr = None
		self.accuracy_ts = None
		self.history = None
		self.batch_size = 40	#2
		self.epochs = 60	#200	#50
		self.dropout = 0.3	#0.5
		self.neurons = 80	#200	#80	#60	#8	#50 #8

		if not action:
			self.find_best_parameters
		else:
			self.train_model
			self.predict

	# funzione che definisce la metrica di r_squared che in keras non e' direttamente disponibile (usata invece nell'SVR con sklearn)
	# def custom_metric_rsquared(self, y_true, y_pred):
	# 	sum_squared_res = K.sum(K.square(y_true - y_pred))
	# 	sum_squared_tot = K.sum(K.square(y_true - K.mean(y_true)))
	# 	return (1 - sum_squared_res / (sum_squared_tot + K.epsilon()))

	# funzione che crea il modello di learning
	# parametri:
	# 	- neurons: neuroni hidden layer
	# 	- dropout: parametro di regolarizzazione
	def create_model(self, neurons=-1, dropout=-1):
		# questo controllo serve per dire che di solito uso i valori di default scelti da me (inizializzati nel costruttore),
		# altrimenti usi i valori passati come parametro
		if neurons == -1:
			neurons = self.neurons
		if dropout == -1:
			dropout = self.dropout
		print 'neurons: ' + str(neurons)
		print 'dropout: ' + str(dropout)

		self.model = Sequential()
		self.model.add(LSTM(self.neurons, input_shape=(self.time_steps, self.x_training_set.shape[2]), dropout=self.dropout))
		if self.output_multi:
			self.model.add(Dense(self.y_test_set.shape[1]))
		else:
			self.model.add(Dense(1))
		self.model.compile(loss='mean_squared_error', optimizer='adam')	#, metrics=[self.custom_metric_rsquared])	Adam(lr=0.01)
		return self.model

	# funzione che decide il numero ottimale di neuroni per l'hidden layer
	def tune_neurons(self):
		neurons = [50, 100, 200]	#[5, 10, 20, 30, 50]
		param_grid = dict(neurons=neurons)
		model = KerasRegressor(build_fn=self.create_model, epochs=self.epochs, batch_size=self.batch_size,
							   verbose=2)
		grid = GridSearchCV(estimator=model, param_grid=param_grid)
		grid_result = grid.fit(self.x_dev_set, self.y_dev_set)
		return grid_result.best_params_

	# funzione che decide il valore ottimale di dropout
	def tune_dropout(self):
		dropout = [0.3, 0.6]	#[0.2, 0.4, 0.6, 0.8]
		param_grid = dict(dropout=dropout)
		model = KerasRegressor(build_fn=self.create_model, epochs=self.epochs, batch_size=self.batch_size,
							   verbose=2)
		grid = GridSearchCV(estimator=model, param_grid=param_grid)
		grid_result = grid.fit(self.x_dev_set, self.y_dev_set)
		return grid_result.best_params_

	# funzione che decide il numero ottimale di epoche per il training e dei mini-batch
	def tune_batch_size_epochs(self):

		batch_size = [2, 30]		#[1, 2, 10, 25, 50]
		epochs = [50, 100]	#[10, 50, 100, 200]
		param_grid = dict(batch_size=batch_size, epochs=epochs)
		model = KerasRegressor(build_fn=self.create_model, epochs=self.epochs, batch_size=self.batch_size,
							   verbose=2)
		grid = GridSearchCV(estimator=model, param_grid=param_grid)
		grid_result = grid.fit(self.x_dev_set, self.y_dev_set)
		return grid_result.best_params_

	@property
	def find_best_parameters(self):
		print "tuning neurons ..."
		param = self.tune_neurons()
		print param
		self.neurons = param['neurons']  # d'ora in poi i modelli che usero' avranno tale numero di neuroni
		print "tuning dropout ..."
		param = self.tune_dropout()
		print param
		self.dropout = param['dropout']  # d'ora in poi i modelli che usero' avranno tale valore di dropout
		print "tuning epochs and mini-batches ..."
		param = self.tune_batch_size_epochs()
		print param
		self.epochs = param['epochs']				# d'ora in poi i modelli che usero' verranno trainati con tali valori di epoche e mini-batch
		self.batch_size = param['batch_size']

	@property
	def train_model(self):
		# creo il modello della LSTM
		self.model = self.create_model()
		# effettuo il training
		self.history = History()  # variabile utile per memorizzare la loss da plottare
		self.model.fit(self.x_training_set, self.y_training_set, epochs = self.epochs, batch_size = self.batch_size, verbose=2, callbacks=[self.history])

		# grafico la loss
		fig = plt.figure(figsize=(12, 10))
		plt.plot(self.history.history["loss"])
		plt.title('LSTM model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.draw()
		#plt.show(block=False)
		fig.savefig("Training_LSTM_loss.png")
		plt.close(fig)
		return self.model

	@property
	def predict(self):
		# effettuo le predizioni
		self.predictions_training = self.model.predict(self.x_training_set)
		self.predictions_test = self.model.predict(self.x_test_set)
		# confronto le predizioni con i targets
		self.score_training = mean_squared_error(self.y_training_set, self.predictions_training) #math.sqrt()
		#self.accuracy_tr = self.model.evaluate(self.x_training_set, self.y_training_set, verbose=0, batch_size=self.batch_size)
		self.accuracy_tr = r2_score(self.y_training_set, self.predictions_training)
		print "Train MSE: " + str(self.score_training) + ", Train R^2: " + str(self.accuracy_tr)
		self.score_test = mean_squared_error(self.y_test_set, self.predictions_test) #math.sqrt()
		#self.accuracy_ts = self.model.evaluate(self.x_test_set, self.y_test_set, verbose=0, batch_size=self.batch_size)
		self.accuracy_ts = r2_score(self.y_test_set, self.predictions_test)
		print "Test MSE: " + str(self.score_test) + ", Test R^2: " + str(self.accuracy_ts)
		return (self.predictions_training, self.predictions_test)