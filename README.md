### Systemy rekomendacyjne
Przykład użycia modelu FFM (implementacja w bibliotece xlearn) do rekomendacji tytułów filmowych z wykorzystaniem bazy danych movielens.

#### Wymagania
- pandas
- numpy
- scikit-learn
- xlearn

#### Test
Po zainstalowaniu bibliotek podanych w wymaganiach możemy uruchomić nasz model za pomocą komendy 
``` bash
> python ffm.py
```
Wynik:
``` bash
pawel@DESKTOP-9TMJ6TT:/mnt/c/Users/pawel/Documents/Projects/xlearn-ffm-rs$ python ffm.py
FFM AUC: 0.7546532978110599
```

#### Omowienie rozwiązania
Zaczynamy od wczytania danych movielens do pandas.dataframe oraz oraz konwersję wartości liczbowych ocen (_rating_) do ich binarnej reprezentacji.
``` python
moviesDF = pd.read_csv('movielens/user_ratedmovies.dat', header=0, delimiter='\t', usecols=['userID', 'movieID', 'rating'], nrows=10000)
moviesDF['rating'] = moviesDF['rating'] > moviesDF['rating'].mean()
```
Następnie możemy wyłączyć oceny do osobnej tablicy jednocześnie zamieniając wartości `True` oraz `False` na odpowiednio 1 oraz 0.
``` python
y = np.where(moviesDF['rating'].values, 1, 0)
```
Model FFM oczekuje natępującego formatu danych wejściowych:
```
<label> <field1>:<feature1>:<value1> <field2>:<feature2>:<value2> ...
.
.
.
```
Kolumna label to będzie nasza tablica ocen stworzona powyżej. `field` będą stanowiły informacje o użytkowniku oraz filmie a w miejsca `feature` wstawimy po prostu identyfikatory liczbowe tychże. Wartość `value` zawsze będzie stanowiła 1 ze względu na to, że obie te informacje są tylko "kategoriami".
``` python
xUserId = np.array(['0:{}:1'.format(i) for i in moviesDF['userID'].values])
xMovieId = np.array(['1:{}:1'.format(i) for i in moviesDF['movieID'].values])
X = np.stack((xUserId, xMovieId), axis=-1)
```
W kolejnej linii dzielimy zestaw danych na testowe oraz te do uczenia modelu.
``` python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
```
Ze względu na specyfikę biblioteki xlearn musimy te dane zapisać do plików (a przynajmniej mi nie udało się uzyskać wyników w inny sposób).
``` python
X_train_transposed = X_train.T
train_output = np.stack((y_train, X_train_transposed[0], X_train_transposed[1]), axis=-1)

with open('tmp/train.dat', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(train_output)

X_val_transposed = X_val.T
test_output = np.stack((y_val, X_val_transposed[0], X_val_transposed[1]), axis=-1)

with open('tmp/test.dat', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(test_output)
```
Kolejna część uruchamia model korzystając z funkcji biblioteki xlearn.
``` python
model = xl.create_ffm()
model.setTrain('tmp/train.dat')
model.setValidate('tmp/test.dat')
model.setTest('tmp/test.dat')

param = {
    'task': 'binary',
    'lr': 0.2,
    'lambda': 0.002,
    'metric': 'acc'
}
model.fit(param, 'tmp/model.dat')

model.setSigmoid()
model.predict('tmp/model.dat', 'tmp/preds.dat')
```
Ponieważ biblioteka zapisuje wynik predykcji na danych testowych do pliku, musimy teraz te dane z powrotem wczytać aby móc wyliczyć metryki.
``` python
with open('tmp/preds.dat', 'r') as f:
    y_pred = np.array([float(i) for i in f.readlines()])
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)
    print('FFM AUC: {}'.format(roc_auc))
```
