# комментарии по русски
# and in english

# A large portion of this originates from
# https://www.wildcardconsulting.dk/useful-information/a-deep-tox21-neural-network-with-rdkit-and-keras/
# Partially converted and heavily commented for use as teaching aid

# требуется для работы с молекулярными отпечатками и дескрипторами
# necessary for working with molecular fingerprints/descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

# практически в любом случае прийдется импортировать эти библиотеки
# numpy - для матриц, pandas - для таблиц
# you will import these in almost every case
# numpy - for matrices, pandas - for tables
import numpy as np
import pandas as pd

# библиотека для визуализаций
# library for visualizations
import matplotlib.pyplot as plt

# один из двух видов структуирования кода нейронной сети с keras
# one of two ways of structuring code using keras
from keras import Sequential

# используемые слои, дропоут - некоторое количество нейронов одного слоя заблокировано
# денс - самые обычные нейроны
# the layers we use. Dropout means a certain percent of the neurons of one layer are blocked from
# signaling to the next layer, Dense is the most standard neuron
from keras.layers import Dropout, Dense

# один из полезных видов метаконтроллеров процесса обучения нейросети
# уменьшает скоростч обучения когда тестовые результаты некоторое время не улучщаются
# one of the useful types of meta control of neural net training
# lowers the learning rate should the testing results cease improving for a while
from keras.callbacks import ReduceLROnPlateau

# оптимизатор - вид градиентного спуска. Обычно используется либо Adam либо RMSProp
# optimizer is the type of gradient descent used. Generally, use Adam or RMSProp
from keras.optimizers import RMSprop

# регуляризаторы позволяшт ограничить расхождение значений в нейросети
# важны для защиты модели от перетренировки
# regularizers prevent the values of a feature of a neuron layer from shooting off into wild values
# are important for protecting the model from overfitting
from keras.regularizers import l1_l2

# самая популярная форма проверки качества классификационной модели
# the most popular approach to checking a classifier model
from sklearn.metrics import roc_auc_score, roc_curve

# для того, чтобы результаты были воспроизводимы,
# set random seed for reproducibility
from numpy.random import seed
seed(8)
from tensorflow import set_random_seed
set_random_seed(8)


# метод для вычисления отпечатков
# method for fingerprint calculation
def morgan_fp(smiles):
	mol = Chem.MolFromSmiles(smiles)
	fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=8192)
	npfp = np.array(list(fp.ToBitString())).astype('int8')
	return npfp


# метод для графической проверки процесса обучения
# если результаты тестовыь данных перестали спускатчся, и пошли вверх, сеть переобучается
# method for visualizing the training process
# if the test loss has ceased going down, and has begun going up, the network is overfitting
def plot_history(history):
	lw = 2
	fig, ax1 = plt.subplots()
	ax1.plot(history.epoch, history.history['binary_crossentropy'], c='b', label="Train", lw=lw)
	ax1.plot(history.epoch, history.history['val_loss'], c='g', label="Val", lw=lw)
	plt.ylim([0.0, max(history.history['binary_crossentropy'])])
	ax1.set_xlabel('Эпох')
	ax1.set_ylabel('Минимизируемое значение')
	ax2 = ax1.twinx()
	ax2.plot(history.epoch, history.history['lr'], c='r', label="Скорость обучения", lw=lw)
	ax2.set_ylabel('Скорость обучения')
	plt.legend()
	plt.show()


# метод для произведения диаграммы результатов, используется метрика площадь под кривой
# чем выше тестовые и валидационные результаты, тем лухше
# method for producing a graph of the results using the metric Area Under Curve
# the higher the test and validation results, the better the model
def show_auc(model):
	pred_train = model.predict(X_train)
	pred_val = model.predict(X_valid)
	pred_test = model.predict(X_test)

	auc_train = roc_auc_score(Y_train, pred_train)
	auc_val = roc_auc_score(Y_valid, pred_val)
	auc_test = roc_auc_score(Y_test, pred_test)
	print("AUC, Тренировка:%0.3F Тестирование:%0.3F Валидация:%0.3F" % (auc_train, auc_test, auc_val))

	fpr_train, tpr_train, _ = roc_curve(Y_train, pred_train)
	fpr_val, tpr_val, _ = roc_curve(Y_valid, pred_val)
	fpr_test, tpr_test, _ = roc_curve(Y_test, pred_test)

	plt.figure()
	lw = 2
	plt.plot(fpr_train, tpr_train, color='b', lw=lw, label='ROC Тренировки (плщд. = %0.2f)' % auc_train)
	plt.plot(fpr_val, tpr_val, color='g', lw=lw, label='ROC Тестирования (плщд. = %0.2f)' % auc_val)
	plt.plot(fpr_test, tpr_test, color='r', lw=lw, label='ROC Валидации (плщд. = %0.2f)' % auc_test)

	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Доля ложноположительных')
	plt.ylabel('Доля истиноположительных')
	plt.title('ROC кривая характеристики %s' % prop)
	plt.legend(loc="lower right")
	plt.show()


# загрузка данных из csv файлож
# load data from csv files
data = pd.read_csv('tox21_train.csv')
testdata = pd.read_csv('tox21_test.csv')
valdata = pd.read_csv('tox21_valid.csv')

# т.к. будем добавлять новую колонку в таблицу с одним из видов отпечатков, присваиваем ей её имя
# since we'll be adding another column to the loaded data with one of the fingerprints, we give that column its name
fp = "morgan"

# итеративно добавляем строчки новой колонки пользуясь ранее написаным методом
# iteratively add rows of the new column using the earlier created method
data[fp] = data["smiles_parent"].apply(morgan_fp)
testdata[fp] = testdata["smiles_parent"].apply(morgan_fp)
valdata[fp] = valdata["smiles_parent"].apply(morgan_fp)

# Выбираем характеристику для моделирования, которые можно увидеть в csv файлах
# Choose property to model, their names can be seen in the csv files
prop = 'SR-MMP'


# оставляем лищь те строчки, у которых для выбранной характеристики есть значения
# leave only the columns that have the selected property
data = data[~data[prop].isnull()]
testdata = testdata[~testdata[prop].isnull()]
valdata = valdata[~valdata[prop].isnull()]

# превращаем таблицу бит векторов/молекулярных отпечатков с начала в список бит векторов,
# затем его в numpy массив для введения в нейросеть
# Convert the molecular fingerprint bit vector table to a bit vector list, then to a numpy array for neural net input
X_train = np.array(list(data[fp]))
X_test = np.array(list(testdata[fp]))
X_valid = np.array(list(valdata[fp]))

# наглядный пример формы вводных данных в простую нейросеть
# example of the shape of data provided as input to a simple neural network
print(np.shape(X_test))

# Собираем данные из колонны таблицы содержащей требуемую характеристику
# Get the values from the column storing the requested characteristic
Y_train = data[prop].values
Y_test = testdata[prop].values
Y_valid = valdata[prop].values

# преобразуем в формат годный для сравнения с выходным результатом нейросети
# reshape into form fit for comparison with neural net output
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
Y_valid = Y_valid.reshape(-1, 1)

# наглядный пример формы выходных данных нейросети
# example of the shape of data used for comparison with neural net output
print(np.shape(Y_test))

# сетевые гиперпараметры (конфигурация и некоторые части архитектуры сети)
# network hyperparameters (configuration/ some parts of architecture)

# коэффициент регуляризации
# regularization coefficient
rgr_val1 = 0.0001
rgr_val2 = 0.016

# процент нейронов одного слоя заблокированных и не влияющих на следующий слой
# какие именно нейроны блокируются рандомно выбирается
# percent of neurons of the previous layer blocked and not passing data to the next
# which specific neurons are blocked is selected randomly
dropout = 0.5

# инициализация регуляризатора
# regularizer initialization
weight_regular = l1_l2(rgr_val1, rgr_val2)

# скорость обучения, сильно влияет на скорость минимизации цележой функции,
# но при высоких значениях неустойчивая и приводит к переобучению либо дестабилизации обучения
# learning rate, strongly affects the speed of the loss minimization
# at high values is unstable, can lead to overfitting, or even destabilization of training
learn_rate = .0002

optim = RMSprop(lr=learn_rate)

# разные функии активации нейрона, один из варьируемых параметров
# different activation functions, one of the parameters of the model you can modify to try and get the best result
activation1 = "relu"
activation2 = "sigmoid"
activation3 = "tanh"


# функция вызываемая в процессе обучения, которая уменьвает скорость обучения на reduction factor если observed metric
# не уменьшилась за последние wait_time количество эпох до минимума minimal_learn_rate
# function called during the training process, that lowers the learning rate by reduction_factor if the observed_metric
# hasn't fallen in the last wait_time epochs

reduction_factor = 0.5
wait_time = 50
minimal_learn_rate = 0.00001
observed_metric = 'val_binary_crossentropy'
reduce_lr = ReduceLROnPlateau(monitor=observed_metric, factor=reduction_factor, patience=wait_time,
                              min_lr=minimal_learn_rate, verbose=1)

#архитектура нейросети
#варьируемые параметры - размер и количесто скрытых слоев, применение dropout слоя,
#выбор между рекуррентной нейронной сетью и стандартной как входной слой
#смотри документацию keras для того, чтобы увидеть все варьируемые параметры

#neural network architecture
#parameters you can modify include the size and number of hidden layers, the use of dropout layers.
#selecting dropout, dense, or a recurrent (LSTM, GRU) layer as the input layer...
#look at keras documentation to get a sense of your options





#инициализация нейросети
#neural net initialization
model = Sequential()

#в данном случае, входной слой является слоем dropout
#можно заменить на dense, GRU, LSTM
model.add(Dropout(0.2, input_shape=(X_train.shape[1],)))

model.add(Dense(output_dim=80, activation=activation1, W_regularizer=weight_regular))
model.add(Dropout(dropout))
model.add(Dense(output_dim=80, activation=activation1, W_regularizer=weight_regular))
model.add(Dropout(dropout))
model.add(Dense(output_dim=80, activation=activation1, W_regularizer=weight_regular))
model.add(Dropout(dropout))

#выходной слой
#output layer
model.add(Dense(Y_train.shape[1], activation='sigmoid'))

# компиляция созданной модели, выбор целевой функции подлежащей минимизации и вида градиентного спуска
# compilation of the created model, selection of the loss function intended to be minimized,
# and type of gradient descent
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['binary_crossentropy'])

# наглядный пример разработанной архитектуры нейросети
# example of produced neural net architecture
print(model.summary())

# одна эпоха - один цикл обучения н-ного колихества данных, разделенных по группам размером на batch_size
# уменьшение batch_size - увеличение точности, но занимает дольше,
# и ужеличение точности после какого то размера не гарантировано
# one epoch - one training cycle on the entire training set, processed in groups of size batch_size
# lower batch size - higher accuracy, but slower, though beyond a certain batch size the effect is negligible
num_of_epochs = 1000
batch_size = 50

# Обучение модели
# Training
history = model.fit(X_train, Y_train, nb_epoch=num_of_epochs, batch_size=batch_size,
                    verbose=2, validation_data=(X_test, Y_test), callbacks=[reduce_lr])

# вызов метода рисующего графу истории обучения
# calling method that draws the training history graph
plot_history(history)

# вызов метода рисующего графу результатов пользуясь метрикой площадь под кривой
# calling the method that draws the results using the AUC as the metric
show_auc(model)