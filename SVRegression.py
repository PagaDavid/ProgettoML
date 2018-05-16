################ SVR ######################
import math
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

class SVRTS:
    def __init__(self, x_training_set, y_training_set, x_dev_set, y_dev_set, x_test_set, y_test_set, action, output_multi):
        self.x_training_set = x_training_set
        self.y_training_set = y_training_set
        self.x_dev_set = x_dev_set
        self.y_dev_set = y_dev_set
        self.x_test_set = x_test_set
        self.y_test_set = y_test_set
        self.action = action        # True per il training, false per la stima dei parametri
        self.output_multi = output_multi  # True se output multiplo, altrimenti False

        self.model = None
        self.predictions_training = None
        self.predictions_test = None
        self.score_training = None
        self.score_test = None
        self.accuracy_tr = None
        self.accuracy_ts = None
        self.C = 1
        self.gamma = 1
        self.epsilon = 0.001

        if not action:
            print self.find_best_parameters
        else:
            self.train_model
            self.predict

    # funzione che crea il modello di learning
    # parametri:
    # 	- C: parametro di trade-off tra generalizzazione e overfitting
    #   - gamma: parametro che modifica la larghezza della gaussiana (nel caso standard di kernel gaussiano 'rbf'')
    def create_model(self, C=-1, gamma=-1, epsilon=-1):
        # questo controllo serve per dire che di solito uso i valori di default scelti da me (inizializzati nel costruttore),
        # altrimenti usi i valori passati come parametro
        if (C == -1):
            C = self.C
        if (gamma == -1):
            gamma = self.gamma
        if (epsilon == -1):
            epsilon = self.epsilon

        self.model = SVR(C=C, gamma=gamma, epsilon=epsilon)
        if (self.output_multi):         # se uso molteplici y, devo fare il wrapping del svr in modo da saperle gestire
            multi_output_model = MultiOutputRegressor(estimator=self.model)
            self.model = multi_output_model

        print self.model
        return self.model


    @property
    def find_best_parameters(self):
        print "Tuning C, gamma and epsilon ..."
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
        gamma_values = [0.001, 0.01, 0.1, 1]
        epsilon_values = [0.001, 0.01, 0.1, 1]

        #self.model = SVR(C=self.C, gamma=self.gamma, epsilon=self.epsilon)
        # in base a quale estimator utilizzo, cambiano i parametri da passare al dizionario per la cv
        if (self.output_multi):
            param_grid = dict(estimator__C=C_values, estimator__gamma=gamma_values, estimator__epsilon=epsilon_values)     # devo aggiungere "estimator__parametro"
        else:
            param_grid = dict(C=C_values, gamma=gamma_values, epsilon=epsilon_values)

        grid_search = GridSearchCV(estimator = self.create_model() , param_grid = param_grid)
        grid_search.fit(self.x_dev_set, self.y_dev_set)

        # in base ai parametri di prima, ottengo gli opportuni valori
        if (self.output_multi):
            self.C = grid_search.best_params_['estimator__C']
            self.gamma = grid_search.best_params_['estimator__gamma']
            self.epsilon = grid_search.best_params_['estimator__epsilon']
        else:
            self.C = grid_search.best_params_['C']
            self.gamma = grid_search.best_params_['gamma']
            self.epsilon = grid_search.best_params_['epsilon']

        return grid_search.best_params_

    @property
    def train_model(self):
        print "Training the model ..."
        # creo il modello del SVR
        self.model = self.create_model()
        # effettuo il training
        self.model.fit(self.x_training_set, self.y_training_set)
        return self.model

    @property
    def predict(self):
        # effettuo le predizioni
        self.predictions_training = self.model.predict(self.x_training_set)
        self.predictions_test = self.model.predict(self.x_test_set)
        # confronto le predizioni con i targets
        self.score_training = mean_squared_error(self.y_training_set, self.predictions_training)    # math.sqrt()
        self.accuracy_tr = self.model.score(self.x_training_set, self.y_training_set)
        print "Train MSE: " + str(self.score_training) + ", Train R^2: " + str(self.accuracy_tr)
        self.score_test = mean_squared_error(self.y_test_set, self.predictions_test)        # math.sqrt()
        self.accuracy_ts = self.model.score(self.x_test_set, self.y_test_set)
        print "Test MSE: " + str(self.score_test) + ", Test R^2: " + str(self.accuracy_ts)
        return (self.predictions_training, self.predictions_test)