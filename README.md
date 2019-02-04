# PyRAO
[![Build Status](https://travis-ci.com/AlexanderBBI144/PyRAO.svg?token=ozxy7p2TNCb5qyxsXwn7&branch=master)](https://travis-ci.com/AlexanderBBI144/PyRAO)
## Описание

Проект состоит из трёх частей: integration, vizualization, analysis.

Integration отвечает за перевод данных обсерватории в международные форматы. Это позволит использовать готовые программные пакеты для анализа данных.

Vizualization - модуль для визуализации данных. Он содержит в себе сервер, ответственный за создание интерактивных графиков на сайте.

Analysis - модуль отвечает за преобразование данных в матрицу объекты-признаки и последующую классификацию этих объектов.

## Установка

Для того, чтобы установить пакет локально, воспользуйтесь pip

```python
pip install pyrao
```

## Использование модуля интеграции

Базовый класс, представляющий полученные с телескопа сырые данные - BSAData.

```python
from pyrao import BSAData

data = BSAData()
```

Для того, чтобы преобразовать данные в формат, распознаваемый сторонними программами необходимо вызвать метод convert. Он принимает на вход список, состоящий из путей к файлам данных наблюдений и калибровочных данных, а также пути к директории, куда требуется записать новый файл. Можно также указать параметры limits* и beams**, которые содержат соответственно диапазон времени наблюдений и лучи радиотелескопа, для которых необходимо выполнить конвертацию.

*Наблюдения отсчитываются от нуля. Каждое наблюдение представляет собой данные, полученные за единицу времени, равную разрешению радиотелескопа (12,5 миллисекунд для больших данных).

**Лучи отсчитываются от нуля вне зависимости от стойки, для которой выполняется конвертация.

```python
path1 = './path/to/obs_N1.pnthr'
path2 = './path/to/obs_N1.txt'
path3 = './path/to/'

data.convert([path1, path2, path3])  # Read and write 48 .fil files from
data.convert([path1, path2, path3], limits=[10, 1000], beams=[4, 9])  # Read and write beams #5 and #10 with observations from 11-th to 1000-th
```

В будущем будет реализована возможность чтения напрямую из базы данных.

Можно проделать действия, которые автоматически выполняет метод convert, самостоятельно. Ниже описываются методы, которые позволят это сделать.

### Чтение данных

Чтобы начать работать с данными, их необходимо прочитать. На текущий момент поддерживается чтение из файлов .pnt, .pnthr.

Для того, чтобы прочитать только часть временного интервала файла, нужно использовать параметр limits метода read. По умолчанию этот параметр равен None (метод читает файл полностью).

```python
data.read(path1, limits=[0, 1000])

data.data  # numpy.ndarray с данными наблюдений
```

Во время чтения происходит автоматическая корректировка лучей, которые были перепутаны с 2014 по 2016 гг.

В будущем возможно будет реализовано чтение файлов напрямую из базы данных.

### Калибровка данных

Сырые данные необходимо откалибровать.

```python
data.calibrate(path2)

data.data  # numpy.ndarray с откалиброванными данными наблюдений
```

В будущем будет реализовано чтение таблицы эквивалентов напрямую из базы данных.

### Запись данных

Для использования данных в дальнейшем анализе с помощью пакета PRESTO, необходимо преобразовать их в формат .fil, который распознаётся большинством нужных нам сторонних пакетов для обработки данных.

Метод write принимает на вход параметр beams, с помощью которого можно указать номера лучей радиотелескопа, которые необходимо записать в файлы. Каждый луч будет записан в отдельном файле. По умолчанию параметр beams равен None (записать все лучи).

```python
data.write(path3, beams=[3], output_type='fil')  # Write .fil with beam #4
data.write(path3, beams=[0, 24], output_type='fil')  # Write .fil with beams #1 and #25
data.write(path3, beams=None, output_type='fil')  # Write 48 separate .fil files
```

В заголовок файла .fil записываются данные о прямом восхождении и склонении на момент начала наблюдений для центральной частоты (формат .fil не позволяет записать координаты для каждой частоты).

В будущем будет реализована запись в формат PSRFITS.

## Модуль визуализации

На текущий момент написаны альфа версии функций визуализации сырых данных. В дальнейшем будет реализована логика серверного приложения, в том числе интеграция с базой данных, а также расширен список поддерживаемых графиков и диаграмм.

## Модуль анализа

Необходимо создать интерфейс, взаимодействующий с программами PRESTO и Pulsar Feature Lab. На вход подается .fil файл, на выход - матрица объекты-признаки.
