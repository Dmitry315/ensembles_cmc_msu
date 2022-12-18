from server_config import *

min_trees = 1
max_trees = 10000
min_features = 0.0
max_features = 1.0
min_d = 1
max_d = 1000
min_lr = 0.0
max_lr = 1.0


class RFForm(FlaskForm):
    model_name = StringField(
        'Название модели',
        [
            validators.InputRequired(message="Введите название модели")
        ]
    )
    n_estimators = DecimalField(
        f'Число деревьев: от {min_trees} до {max_trees}',
        [
            validators.InputRequired(message="Введите число деревьев"),
            validators.NumberRange(min=min_trees, max=max_trees,
                                   message="Число деревьев - натуральное число не превыщающее %(max)d.")
        ]
    )
    feature_subsample_size = FloatField(
        f'Доля признаков для обучения от {min_features} до {max_features}',
        [
            validators.InputRequired(message="Введите доля признаков"),
            validators.NumberRange(min=min_features, max=max_features,
                                   message="Доля признаков - вещественное число от %(min)d до %(max)d.")
        ]
    )
    max_depth = DecimalField(
        f'Максимальная глубина деревьев от {min_d} до {max_d}',
        [
            validators.InputRequired(message="Введите число деревьев"),
            validators.NumberRange(min=min_d, max=max_d,
                                   message="Максимальная глубина - натуральное число не превыщающее %(max)d.")
        ]
    )
    file = FileField(
        'Файл для тренировки модели',
        [
            validators.DataRequired(message="Для обучения необходим файл"),
            FileAllowed(['csv'], message='Можно загружать только csv файлы')
        ])
    val_file = FileField(
        'Файл для валидации модели (не обязателен)',
        [
            FileAllowed(['csv'], message='Можно загружать только csv файлы')
        ])
    target = StringField(
        'Метка для предсказания (target)',
        [
            validators.InputRequired(message="Введите название метки")
        ]
    )
    submit = SubmitField('Иницализировать модель')


class GBForm(FlaskForm):
    model_name = StringField(
        'Название модели',
        [
            validators.InputRequired(message="Введите название модели")
        ]
    )
    n_estimators = DecimalField(
        f'Число деревьев от {min_trees} до {max_trees}',
        [
            validators.InputRequired(message="Введите число деревьев"),
            validators.NumberRange(min=min_trees, max=max_trees,
                                   message="Число деревьев - натуральное число не превыщающее %(max)d.")
        ]
    )
    max_depth = DecimalField(
        f'Максимальная глубина деревьев от {min_d} до {max_d}',
        [
            validators.InputRequired(message="Введите число деревьев"),
            validators.NumberRange(min=min_d, max=max_d,
                                   message="Максимальная глубина - натуральное число не превыщающее %(max)d.")
        ]
    )
    feature_subsample_size = FloatField(
        f'Доля признаков для обучения от {min_features} до {max_features}',
        [
            validators.InputRequired(message="Введите доля признаков"),
            validators.NumberRange(min=min_features, max=max_features,
                                   message="Доля признаков - вещественное число от %(min)d до %(max)d.")
        ]
    )
    learning_rate = FloatField(
        f'Скорость обучения от {min_lr} до {max_lr}',
        [
            validators.InputRequired(message="Введите доля признаков"),
            validators.NumberRange(min=min_features, max=max_features,
                                   message="Доля признаков - вещественное число от %(min)d до %(max)d.")
        ]
    )
    file = FileField(
        'Файл для тренировки модели',
        [
            validators.DataRequired(message="Для обучения необходим файл"),
            FileAllowed(['csv'], message='Можно загружать только csv файлы')
        ])
    val_file = FileField(
        'Файл для валидации модели (не обязателен)',
        [
            FileAllowed(['csv'], message='Можно загружать только csv файлы')
        ])
    target = StringField(
        'Метка для предсказания (target)',
        [
            validators.InputRequired(message="Введите название метки")
        ]
    )
    submit = SubmitField('Иницализировать модель')


class TestForm(FlaskForm):
    file = FileField(
        'Файл для тестирования модели',
        [
            validators.DataRequired(message="Для предсказания необходим файл"),
            FileAllowed(['csv'], message='Можно загружать только csv файлы')
        ])
    submit = SubmitField('Сделать предсказание')


def checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except Exception as err:
            app.logger.info(str(err))
            for f in files_to_delete:
                try:
                    os.remove(f)
                except Exception:
                    pass
                files_to_delete.remove(f)
            abort(418)

    return wrapper


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


@app.errorhandler(418)
def page_not_found(e):
    return render_template('418.html')


def dump_model(model, path):
    with open(path, 'wb') as out_f:
        pickle.dump(model, out_f, pickle.HIGHEST_PROTOCOL)


def load_model(path):
    with open(path, 'rb') as in_f:
        model = pickle.load(in_f)
    return model


def fit_model(model, file_path, target, val_file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target])
    obj_columns = X.columns[X.dtypes == object]
    X = X.drop(columns=obj_columns)
    if 'id' in X.columns:
        X = X.drop(columns=['id'])
    y = df[target]
    X_val = None
    y_val = None
    if val_file_path:
        val_df = pd.read_csv(val_file_path)
        X_val = val_df[X.columns].values
        y_val = val_df[target].values
    model.fit(X.values, y.values, X_val, y_val)
    return X.columns


def fit_save_rf_model(form, request, model_idx):
    model_name = request.form["model_name"]
    target = request.form["target"]
    n_estimators = int(request.form["n_estimators"])
    max_depth = int(request.form["max_depth"])
    feature_subsample_size = float(request.form["feature_subsample_size"])
    filename = secure_filename(form.file.data.filename)
    val_filename = secure_filename(form.val_file.data.filename)
    if val_filename == '':
        val_filename = None
    val_files_path = None
    if val_filename:
        val_files_path = app.config['UPLOAD_FOLDER'] + filename
    files_path = app.config['UPLOAD_FOLDER'] + filename
    form.file.data.save(files_path)
    if val_filename:
        form.val_file.data.save(val_files_path)
        files_to_delete.append(val_files_path)
    files_to_delete.append(files_path)

    model = RandomForestMSE(n_estimators=n_estimators, max_depth=max_depth,
                            feature_subsample_size=feature_subsample_size)
    features = fit_model(model, files_path, target, val_files_path)
    files_to_delete.remove(files_path)
    if val_filename:
        files_to_delete.remove(val_files_path)
    model_path = app.config['MODELS_FOLDER'] + f'{model_idx}.pkl'
    dump_model(model, model_path)
    model_params[model_idx] = {
        'type': 'rf',
        'name': model_name,
        'target': target,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'feature_subsample_size': feature_subsample_size,
        'path': model_path,
        'training_file': filename,
        'features': features
    }
    if val_filename:
        model_params[model_idx]['val_file'] = val_filename


def fit_save_gb_model(form, request, model_idx):
    model_name = request.form["model_name"]
    target = request.form["target"]
    n_estimators = int(request.form["n_estimators"])
    max_depth = int(request.form["max_depth"])
    feature_subsample_size = float(request.form["feature_subsample_size"])
    learning_rate = float(request.form["learning_rate"])
    filename = secure_filename(form.file.data.filename)
    val_filename = secure_filename(form.val_file.data.filename)
    if val_filename == '':
        val_filename = None
    val_files_path = None
    if val_filename:
        val_files_path = app.config['UPLOAD_FOLDER'] + filename
    files_path = app.config['UPLOAD_FOLDER'] + filename
    form.file.data.save(files_path)
    if val_filename:
        form.val_file.data.save(val_files_path)
        files_to_delete.append(val_files_path)
    files_to_delete.append(files_path)

    model = GradientBoostingMSE(n_estimators=n_estimators, max_depth=max_depth,
                                feature_subsample_size=feature_subsample_size,
                                learning_rate=learning_rate)
    features = fit_model(model, files_path, target, val_files_path)
    files_to_delete.remove(files_path)
    if val_filename:
        files_to_delete.remove(val_files_path)
    model_path = app.config['MODELS_FOLDER'] + f'{model_idx}.pkl'
    dump_model(model, model_path)
    model_params[model_idx] = {
        'type': 'gb',
        'name': model_name,
        'target': target,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'feature_subsample_size': feature_subsample_size,
        'learning_rate': learning_rate,
        'path': model_path,
        'training_file': filename,
        'features': features
    }
    if val_filename:
        model_params[model_idx]['val_file'] = val_filename


@app.route('/')
@app.route('/home')
@checker
def home():
    return render_template('home.html')


@app.route('/models')
@checker
def models():
    return render_template('models.html', models=model_params)


@app.route('/models/<int:idx>/edit', methods=['GET', 'POST'])
@checker
def edit_model(idx):
    if idx not in model_params.keys():
        abort(404)
    params = model_params[idx]
    model_type = params['type']
    if model_type == 'rf':
        form = RFForm()
        if form.validate_on_submit():
            fit_save_rf_model(form, request, idx)
            return redirect(f'/models/{idx}')
        return render_template('random_forest.html', form=form, params=params)
    if model_type == 'gb':
        form = GBForm()
        if form.validate_on_submit():
            fit_save_gb_model(form, request, idx)
            return redirect(f'/models/{idx}')
        return render_template('grad_boost.html', form=form, params=params)
    abort(404)


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download_file(filename):
    try:
        directory = os.path.abspath(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(directory, secure_filename(filename))
        return send_file(file_path)
    except Exception as err:
        app.logger.info(str(err))
        abort(404)


@app.route('/models/<int:idx>', methods=['GET', 'POST'])
@checker
def get_model(idx):
    form = TestForm()
    if idx not in model_params.keys():
        abort(404)
    params = model_params[idx]
    model = load_model(params['path'])
    history = model.train_rmse_history
    df = pd.DataFrame(history, index=range(len(history)), columns=['RMSE'])
    df['Номер итерации'] = range(1, len(history) + 1)
    df['Выборка'] = 'Тренировочная'
    if 'val_file' in params.keys():
        val_history = model.val_rmse_history
        tmp = pd.DataFrame(val_history, index=range(len(val_history)), columns=['RMSE'])
        tmp['Номер итерации'] = range(1, len(history) + 1)
        tmp['Выборка'] = 'Валидационная'
        df = pd.concat((df, tmp))
    fig = px.line(df, x='Номер итерации', y='RMSE', color='Выборка',
                  title='Зависимость RMSE от номера итерации')

    graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        files_path = app.config['UPLOAD_FOLDER'] + filename
        form.file.data.save(files_path)
        files_to_delete.append(files_path)
        # predict
        features = params['features']
        target = params['target']
        df = pd.read_csv(files_path)
        X = df[features]
        pred = pd.DataFrame(model.predict(X.values), index=X.index, columns=[target])
        pred.to_csv(app.config['UPLOAD_FOLDER'] + 'predict.csv')
        return redirect(url_for('download_file', filename='predict.csv'))
    return render_template('get_model.html', params=params, idx=idx, graph=graph, form=form)


@app.route('/random_forest', methods=['GET', 'POST'])
@checker
def random_forest():
    form = RFForm()
    if request.method == 'POST' and form.validate_on_submit():
        model_idx = app.config['MODEL_IDX']
        app.config['MODEL_IDX'] += 1
        fit_save_rf_model(form, request, model_idx)
        return redirect(url_for('models'))
    return render_template('random_forest.html', form=form)


@app.route('/gradient_boosting', methods=['GET', 'POST'])
@checker
def grad_boost():
    form = GBForm()
    if request.method == 'POST' and form.validate_on_submit():
        model_idx = app.config['MODEL_IDX']
        app.config['MODEL_IDX'] += 1
        fit_save_gb_model(form, request, model_idx)

        return redirect(url_for('models'))
    return render_template('grad_boost.html', form=form)


@app.route('/about')
@checker
def about():
    return render_template('about.html')


if __name__ == '__main__':
    # run server
    app.run(host="0.0.0.0", port="8080")
