# conll2vec
conll2vec — утилита для построения векторных представлений слов на основе морфологически и синтаксически размеченных данных в формате [conll](https://universaldependencies.org/format.html).

В ходе обучения conll2vec одновременно оперирует синтаксическим и оконным контекстом. Часть измерений векторного представления обучается на синтаксических контекстах, часть — на традиционных для [word2vec](https://ru.wikipedia.org/wiki/Word2vec)-утилит линейно-оконных контекстах. 
* Синтаксический контекст слова определяется по дереву зависимостей. Для каждого слова отыскиваются его ближайшие соседи в синтаксическом дереве (родитель и непосредственные потомки). Контекстными единицами выступают тройки <лемма слова-соседа, направление синтаксической связи, тип синтаксического отношения>. 
* Для линейно-оконных контекстов размер окна приравнивается размеру предложения. Контекстными единицами выступают леммы слов.

Утилита строит отдельные векторные представления для нарицательных и собственных имён. Нарицательные *клин*, *орёл*, *курган*, *газ*, *калина*, *сапсан*, *барак* и многие другие имеют омонимы среди собственных имён. Морфологическая информация в conll-данных позволяет различать такого рода омонимы и создавать для них отдельные представления.

Построение векторных представлений утилитой conll2vec выполняется в несколько этапов, некоторые из которых опциональны (необходимость их выполнения зависит от назначения векторной модели). Полный перечень включает:
* предобработку conll-данных [опционально];
* построение словарей;
* построение модели лемм без собственных имён;
* построение полной модели лемм [опционально];
* построение модели токенов [опционально];
* добавление знаков препинания [опционально].

Кроме того, в утилиту встроены средства оценки построенной модели на стандартных тестсетах (для русскоязычных моделей) и вычисления семантической близости лексических единиц (в пакетном и интерактивном режиме).

## Быстрый старт
В репозитории размещены демонстрационные скрипты для Linux и Windows, обеспечивающие:
1. сборку утилиты из исходных кодов, 
2. построение векторной модели на основе фрагмента данных из корпуса [PaRuS](https://parus-proj.github.io/PaRuS),
3. запуск утилиты в интерактивном режиме — для заданного слова отыскиваются ближайшие по смыслу.

Для сборки утилиты требуется компилятор с поддержкой [C++17](https://ru.wikipedia.org/wiki/C%2B%2B17) и библиотека [ICU](http://site.icu-project.org/home). Сборка протестирована под Linux с компилятором gcc v9.3.0 и под Windows с компилятором от Visual Studio 2017 v15.9.18 (cl.exe версии 19.16).

<table>
  <tr>
    <th width="50%">Запуск под Linux</th>
    <th>Запуск под Windows</th>
  </tr>
  <tr>
    <td valign="top">Запустите консоль. Перейдите в директорию, в которой хотите развернуть программное обеспечение.</td>
    <td>Запустите «Командную строку Native Tools x64 для VS 2017» (это обеспечит настройку окружения для сборки утилит). Перейдите в папку, в которой хотите развернуть программное обеспечение.</td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <em>Примечание</em>: для загрузки обучающего дейтасета необходим <a href="https://git-lfs.github.com/">git-lfs</a>.<br/>
      git clone https://github.com/parus-proj/conll2vec.git<br/>
      cd conll2vec
    </td>
  </tr>
  <tr>
    <td valign="top">Установите библиотеку ICU.<br/>sudo apt install libicu</td>
    <td>Установите библиотеку ICU.<br/>Воспользуйтесь скриптом icu_install.cmd для установки 64-битной версии библиотеки для msvc2017 непосредственно в каталог утилиты conll2vec.<br/>&nbsp;&nbsp;<i>или</i><br/>Установите ICU, следуя инструкции по установке на сайте библиотеки. Затем в открытом окне командной строки для VS 2017 настройте переменную окружения PATH так, чтобы в ней присутствовал каталог с dll-библиотеками ICU, а в файле makefile.msvc определите переменную ICU_PATH так, чтобы она указывала на корневой каталог библиотеки (внутри этого каталога должны находиться каталоги include и lib64).</td>
  </tr>
  <tr>
    <td align="center">./demo-linux.sh</td>
    <td align="center">demo-windows.cmd</td>
  </tr>
  <tr>
    <td colspan="2">При успешных сборке и обучении работа скрипта завершится вызовом утилиты в режиме поиска слов с близким значением. Чтобы удостовериться в работоспособности, попробуйте ввести распространённые слова — <i>президент</i>, <i>комната</i>, <i>автомобиль</i>. Отмечу, что демонстрационный скрипт порождает неоптимальную векторную модель, чтобы сократить время обучения. Для построения высококачественной модели необходимо использовать больший объём данных из корпуса <a href="https://parus-proj.github.io/PaRuS">PaRuS</a> или иного крупного conll-дейтасета.</td>
  </tr>
</table>

## Режимы работы и параметры утилиты

В conll2vec реализованы алгоритмы для решения нескольких задач. Выбор задачи осуществляется параметром `-task`. К числу основных задач относятся следующие.
* vocab — режим построения словарей по conll-дейтасету. В этом режиме утилита по обучающим данным порождает пять словарных ресурсов:
    + основной словарь лемм,
    + словарь лемм для собственных имён,
    + словарь синтаксических контекстов,
    + словарь словоформ (токенов),
    + отображение из лемм в соответствующие возможные словоформы.
* train — режим обучения модели по conll-дейтасету. Типично применяется два раза: сначала для обучения векторным представлениям основного словаря, затем модель расширяется собственными именами.
* sim — интерактивный режим, позволяющий для заданного слова находить ближайшие по смыслу (используется [косинусная мера](https://en.wikipedia.org/wiki/Cosine_similarity)). Это расширенный аналог утилиты distance из оригинального word2vec.

Основными параметрами при построении словарей являются `-train`, задающий имя файла с обучающими conll-данными, а также группа параметров, задающих имена файлов для сохранения словарей: `-vocab_m` (основной словарь лемм), `-vocab_p` (словарь собственных имён), `-vocab_d` (словарь синтаксических контекстов), `-vocab_t` (словарь словоформ) и `-tl_map` (отображение леммы–токены). Частотные пороги для включения слова/контекста в словарь возможно определять отдельно для каждого словаря. За это отвечает группа параметров `-min-count_m`, `-min-count_p`, `-min-count_d`, `-min-count_t` (это аналоги параметра `-min-count` в word2vec).

Ниже приводится пример командной строки для запуска conll2vec в режиме построения словарей:

```
./conll2vec -task vocab -train data.conll \
            -vocab_m main.vocab -vocab_p proper_names.vocab -vocab_d dep_ctx.vocab -vocab_t tokens.vocab -tl_map l2t.map \
            -min-count_m 70 -min-count_p 100 -min-count_d 50 -min-count_t 50
```

Построение векторных представлений выполняется в соответствии с архитектурой skip-gram и подходом к снижению вычислительной нагрузки negative sampling. Сначала векторные представления строятся для основного словаря лемм — в режиме train указывается параметр `-vocab_m`. Обученная векторная модель сохраняется в файл, заданный параметром `-model`. Если в дальнейшем требуется дополнить модель собственными именами, то при обучении основного словаря необходимо также указать параметр `-backup`. Он позволяет сохранить весовые матрицы нейросети в файл. Для дообучения модели (т. е. дополнения её собственными именами) утилиту необходимо запустить ещё раз, указав параметры `-vocab_p` и `-restore` (вместо `-vocab_m` и `-backup`).

Кроме того, для обучения утилите необходимо знать имя файла с обучающими conll-данными (параметр `-train`), имена файлов со словарями контекстов (`-vocab_d` и `-vocab_a`; последний в настоящей реализации должен совпадать с именем основного словаря лемм), размерности частей векторного представления, обучаемых с учётом синтаксических и линейно-оконных контекстов (`-size_d` и `-size_a`). Сумма последних двух параметров даёт итоговую размерность векторных представлений модели.

Параметры позволяют также задать количество эпох обучения (`-iter`), начальное значение коэффициента скорости обучения (`-alpha`), количество отрицательных примеров, приходящихся на один положительный (`-negative`), коэффициенты субдискретизации (`-sample_w`, `-sample_d`, `-sample_a` — аналоги параметра `-sample` в word2vec для основного и двух контекстных словарей), количество потоков (`-threads`).

Ниже приводятся примеры команд для запуска conll2vec в режиме обучения векторной модели (сначала по основному словарю, потом дообучение собственным именам):

```
./conll2vec -task train -train data.conll \
            -vocab_m main.vocab -backup backup.data -vocab_d dep_ctx.vocab -vocab_a main.vocab \
            -model vectors.bin -size_d 75 -size_a 25
            
./conll2vec -task train -train data.conll \
            -vocab_p proper_names.vocab -restore backup.data -vocab_d dep_ctx.vocab -vocab_a main.vocab \
            -model vectors.bin -size_d 75 -size_a 25
```
Запуск conll2vec в интерактивном режиме для поиска близких по значению слов требует указания параметров, определяющих имя файла с сохранённой векторной моделью (`-model`) и размерности частей векторного представления (`-size_d` и `-size_a`).

Пример команды:

```
./conll2vec -task sim -model vectors.bin -size_d 75 -size_a 25 -size_g 0
```

## Специальные режимы работы

Кроме трёх основных задач, о которых шла речь выше, утилита conll2vec может выполнять различные преобразования тренировочных данных и построенной модели. Рассмотрим расширенный набор задач (значений для параметра -task).
* fit — вспомогательный режим преобразования conll-дейтасетов, выбранных из корпуса [PaRuS](https://parus-proj.github.io/PaRuS) для повышения качества векторных представлений и ускорения обучения. Утилита фильтрует малозначимые синтаксические связи, строит связи в обход служебных текстовых единиц, обобщает числовые величины, приводит к нижнему регистру словоформы и др.
* unPNize — режим смешивания собственных имён с остальной лексикой. В ходе обучения собственные имена добавляются в словарь модели со служебным суффиксом _PN. Это сделано для различения омонимов в парах нарицательное-собственное (*клин*, *орел* и др.). Таким образом, в модели собственные имена изначально имеют вид *китай_PN*, *иван_PN* и т. п. Режим смешивания позволяет избавиться от суффикса. Для омонимичных слов векторное представление находится как взвешенное среднее омонимов (веса вычисляются на основе частот каждого из омонимов).
* toks — режим добавления словоформ в модель. Информация о соответствии словоформ леммам берётся из словаря, указываемого параметром `-tl_map`. Если словоформе соответствует единственная лемма, то вектор для словоформы порождается в ближайшей окрестности вектора леммы (выполняется небольшое случайное смещение относительно леммы). В случае [омоформии](https://ru.wikipedia.org/wiki/%D0%9E%D0%BC%D0%BE%D0%BD%D0%B8%D0%BC%D1%8B#%D0%9E%D0%BC%D0%BE%D0%BD%D0%B8%D0%BC%D1%8B,_%D0%BE%D0%BC%D0%BE%D1%84%D0%BE%D0%BD%D1%8B,_%D0%BE%D0%BC%D0%BE%D0%B3%D1%80%D0%B0%D1%84%D1%8B_%D0%B8_%D0%BE%D0%BC%D0%BE%D1%84%D0%BE%D1%80%D0%BC%D1%8B) результирующий вектор для словоформы находится как взвешенное среднее векторов его возможных лемм (веса вычисляются на основе частот в корпусе).
* toks_train — \[экспериментальный\] режим доучивания модели после добавления в неё словоформ.
* toks_gramm — \[экспериментальный\] режим доучивания модели словоформ грамматическим признакам. В результате к уже построенному векторному представлению слова будет добавлен вектор, кодирующий близость по грамматическим характеристикам. Такие вектора полезны в задачах синтеза текста и морфологического анализа. Длина векторов модели увеличивается на величину параметра `-size_g`.
* punct — режим добавления в векторную модель знаков пунктуации (они не включаются в основной словарь). Вектора для них порождаются эвристическим алгоритмом.
* balance — режим масштабирования группы измерений, обучавшихся на линейно-оконных контекстах (отвечающих за ассоциативную близость значений слов).  Масштабирующий коэффициент задаётся параметром `-a_ratio`. Если он больше единицы, то «ассоциативные» измерения будут вносить больший вклад в результирующую меру близости, вычисляемую по всему вектору. Если коэффициент между 0 и 1, то вклад этих измерений в величину близости снижается.
* sub — режим извлечения подмодели, заданной диапазоном измерений. Например, можно из обученной модели извлечь подмодель, отвечающую только за категориальную близость или только за ассоциации между значениями.
* fsim — пакетный режим вычисления близости значений для пар слов. Типичный сценарий применения — оценка качества модели.
* selftest_ru — режим самодиагностики и оценки (для русского языка). В данном режиме утилита выполняет проверку модели, а также вычисляет показатели качества модели на стандартных тестовых множествах для определения семантической близости ([RUSSE'15](http://www.dialog-21.ru/media/1123/panchenkoaetal.pdf) и [RuSim1000](http://www.dialog-21.ru/media/3958/trofimovivsuleymanovaea.pdf)).

Примеры запуска утилиты conll2vec в различных режимах.

```
# fit для случая неархивированных обучающих данных. Результат преобразования помещается в файл, задаваемый параметром -train.
./conll2vec -task fit -fit_input parus.conll -train data.conll

# Т.к. объём обучающих данных обычно большой, удобнее использовать вариант fit в конвейерном исполнении. Например, так.
bzip2 -dkc parus.conll.bz2 | ./conll2vec -task fit -fit_input stdin -train data.conll

# Смешивание собственных имён с основной лексикой.
./conll2vec -task unPNize -model vectors.bin -vocab_m main.vocab -vocab_p proper_names.vocab

# Добавление словоформ в модель.
./conll2vec -task toks -model vectors.bin -tl_map l2t.map

# Доучивание словоформ (экспериментальное!).
./conll2vec -task toks_train -train data.conll \
            -vocab_t tokens.vocab -restore backup.data -vocab_d dep_ctx.vocab -vocab_a main.vocab \
            -model vectors.bin -size_d 75 -size_a 25

# Доучивание грамматических признаков (экспериментальное!).
./conll2vec -task toks_gramm -train data.conll -vocab_t tokens.vocab -model vectors.bin \
            -size_d 75 -size_a 25 -size_g 20
            
# Добавление знаков препинания в модель.
./conll2vec -task punct -model vectors.bin

# Масштабирование группы измерений, обучавшихся на линейно-оконных контекстах.
./conll2vec -task balance -model vectors.bin -size_d 75 -size_a 25 -a_ratio 4.0

# Извлечение подмодели.
./conll2vec -task sub -model vectors.bin -sub_l 0 -sub_r 75

# Пакетный режим вычисления близости значений слов.
# Пары слов в файле word_pairs.csv записываются в формате:
#   слово1,слово2,
# Результат выводится в стандартный поток вывода в формате:
#   слово1,слово2,близость_по_всему_вектору,близость_по_синтаксич_части,близость_по_оконной_части
./conll2vec -task fsim -model vectors.bin -size_d 75 -size_a 25 -fsim_file word_pairs.csv -fsim_fmt detail

# Самодиагностика.
./conll2vec -task selftest_ru -model vectors.bin -model_fmt bin -size_d 75 -size_a 25 -size_g 0 -a_ratio 1.0 -st_yo 0
```

## Благодарности

Утилита conll2vec создана при финансовой поддержке РФФИ в рамках научного проекта № 19-07-00779.
