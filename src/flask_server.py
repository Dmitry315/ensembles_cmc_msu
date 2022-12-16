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
    target = StringField(
        'Метка для предсказания (target)',
        [
            validators.InputRequired(message="Введите название метки")
        ]
    )
    submit = SubmitField('Иницализировать модель')


def checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            res = func()
            return res
        except:
            abort(418)

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


def fit_model(model, file_path, target):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target])
    obj_columns = X.columns[X.dtypes == object]
    X = X.drop(columns=obj_columns)
    if 'id' in X.columns:
        X = X.drop(columns=['id'])
    y = df[target]
    model.fit(X.values, y.values)
    return X.columns


def fit_save_rf_model(form, request, model_idx):
    model_name = request.form["model_name"]
    target = request.form["target"]
    n_estimators = int(request.form["n_estimators"])
    max_depth = int(request.form["max_depth"])
    feature_subsample_size = float(request.form["feature_subsample_size"])
    filename = secure_filename(form.file.data.filename)
    files_path = app.config['UPLOAD_FOLDER'] + filename
    form.file.data.save(files_path)

    model = RandomForestMSE(n_estimators=n_estimators, max_depth=max_depth,
                            feature_subsample_size=feature_subsample_size)
    features = fit_model(model, files_path, target)
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
        'training_file': files_path,
        'features': features
    }


def fit_save_gb_model(form, request, model_idx):
    model_name = request.form["model_name"]
    target = request.form["target"]
    n_estimators = int(request.form["n_estimators"])
    max_depth = int(request.form["max_depth"])
    feature_subsample_size = float(request.form["feature_subsample_size"])
    learning_rate = float(request.form["learning_rate"])
    filename = secure_filename(form.file.data.filename)
    files_path = app.config['UPLOAD_FOLDER'] + filename
    form.file.data.save(files_path)

    model = GradientBoostingMSE(n_estimators=n_estimators, max_depth=max_depth,
                                feature_subsample_size=feature_subsample_size,
                                learning_rate=learning_rate)
    features = fit_model(model, files_path, target)
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
        'training_file': files_path,
        'features': features
    }


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/models')
def models():
    return render_template('models.html', models=model_params)


@app.route('/models/<int:idx>/edit', methods=['GET', 'POST'])
def edit_model(idx):
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

@app.route('/models/<int:idx>', methods=['GET', 'POST'])
def get_model(idx):

    params = model_params[idx]
    return render_template('get_model.html', params=params, idx=idx)


@app.route('/random_forest', methods=['GET', 'POST'])
def random_forest():
    form = RFForm()
    if request.method == 'POST' and form.validate_on_submit():
        model_idx = app.config['MODEL_IDX']
        app.config['MODEL_IDX'] += 1
        fit_save_rf_model(form, request, model_idx)

        return redirect(url_for('models'))
    return render_template('random_forest.html', form=form)


@app.route('/gradient_boosting', methods=['GET', 'POST'])
def grad_boost():
    form = GBForm()
    if request.method == 'POST' and form.validate_on_submit():
        model_idx = app.config['MODEL_IDX']
        app.config['MODEL_IDX'] += 1
        fit_save_gb_model(form, request, model_idx)

        return redirect(url_for('models'))
    return render_template('grad_boost.html', form=form)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    # run server
    # app.run()
    app.run(port='8000', host='127.0.0.1', debug=True)
