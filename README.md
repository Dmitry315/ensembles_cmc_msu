<h1> Ансамбли алгоритмов. Веб-сервер. Композиции алгоритмов для решения задачи регрессии. </h1>
<h2>Алгоритмы ансамблирования</h2>
Алгоритмы реализованы в src/ensembles.py
<h3>Random Forest</h3>
класс RandomForestMSE - реализует алгоритм случайного леса с квадратичной функцией потерь.<br>
fit(X, y, X_val, y_val)<br>
predict(X, y) -> pred
<h3>Gradient Boosting</h3>
класс GradientBoostingMSE - реализует алгоритм градиентного бустинга над решающими деревьями с квадратичной функцией потерь.<br>
fit(X, y, X_val, y_val)<br>
predict(X, y) -> pred
<h2>Работа с приложением</h2>
Готовый докер образ: https://hub.docker.com/repository/docker/dmitry315/ensembles_server/general <br>

1) Если возникли проблемы c разрешением 

```bash
chmod +x scripts/build.sh
chmod +x scripts/run.sh
```

2) Собрать образ докера (запускать из ensembles_cmc_msu):

```bash
scripts/build.sh
```

3) Запустить приложение (запускать из ensembles_cmc_msu):

```bash
scripts/run.sh
```

<h2> Файлы </h2>
<p>src - исходники с веб-приложением и алгоритмами.</p>
<p>scripts - скрипты для сборки и запуска контейнера docker</p>

