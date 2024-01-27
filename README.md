# convolutional-neural-networks
**Компьютерное зрение**

**Что такое компьютерное зрение?**

**Компьютерное зрение** – это область искусственного интеллекта, связанная с созданием компьютерных систем, которые способны получать полезную информацию с изображений и видео.

**Какие задачи решает компьютерное зрение?**

1. **Распознавание**. Классическая задача компьютерного зрения. Она позволяет распознавать, детектировать, а также сегментировать объекты и образы на изображении. Примером такой задачи может быть поиск изображений по содержанию, оценка положения объекта на изображении.
1. **Движение**. Задача связана с оценкой движения объекта. Обычно для решения такой задачи используют некоторую последовательность картинок (видео). В результате мы можем получить информацию о скорости объекта, направлении его движения, также можно получить положение самой камеры в пространстве.
1. **Восстановление сцены**. В простейшем случае программа по одному или нескольким изображениям может создать набор точек трёхмерного изображения. Более сложные программы создают полную трехмерную модель.
1. **Восстановление изображений**. В первую очередь – это удаление шума, размытости или искажения с изображения. Простым подходом к решению такой задачи являются различные типы фильтров, а более сложным методом используют представления того, как должны выглядеть те или иные участки изображения. Существует даже полная генерация изображения.

**Где используется компьютерное зрение?**

Применение компьютерного зрения:

1. **Безопасность**. Это распознавание лиц (Face ID), камеры видеонаблюдения на дорогах.
1. **Промышленность**. Прикладная область компьютерного зрения. Для промышленности используют отдельный термин – Машинное зрение. Примером может служить контроль качества, когда детали или конечный продукт автоматически проверяются на наличие дефектов, а также управление определёнными манипуляторами.
1. **Медицина**. Компьютерное зрение помогает найти закономерности и аномалии в МРТ-, КТ- и рентгеновских снимках, экономит время на интерпретации результатов исследования и позволяет врачам получить дополнительную информацию о патологии. Диагнозы ставятся точнее и быстрее.
1. **В приложениях и телефонах**. Современные поисковики, камеры, а также переводчики позволяют работать с изображениями, а с помощью Face ID можно разблокировать свой телефон по лицу.

Сегодняшние тренды компьютерного зрения:

1. **Беспилотные автомобили**. Нейросети способны управлять транспортом без участия человека. Яндекс активно развивает такие технологии. Уже сегодня в некоторых районах Москвы можно заказать беспилотное такси.
1. Нейросети позволяют **генерировать изображения**. По тексту можно сгенерировать целое изображение. Даже Photoshop разрабатывает новые инструменты, которые позволяют дополнять или перерисовывать участки изображения.

**Как работает компьютерное зрение?**

Работу компьютерного зрения можно разделить на три этапа: захват изображения, его обработка и какая-либо операция. Для захвата используют фоточувствительные датчики, цифровые камеры, ультрафиолетовые или инфракрасные камеры. Эти устройства захватывают изображения и преобразуют их в цифровую информацию.

В основе работы большинства систем компьютерного зрения лежит **глубокое обучение**.

**Глубокое обучение** – это подмножество **машинного обучения**, в котором используются специальные алгоритмические структуры. Глубокое обучение позволяет работать с большими объемами данных, более точно находить признаки при обучении и позволяет использовать разные типы данных (например, картинки или звук).

В компьютерном зрении используются **сверточные нейронные сети**.

Основная задача сверточной нейронной сети – это классификация изображений. При использовании сверточной нейронной сети можно существенно уменьшить количество обучаемых параметров и получать высокое качество классификации.

**Как работает сверточная нейронная сеть?**

**Сверточные нейронные сети** – это специальная архитектура искусственных нейронных сетей, предложенная Яном Лекуном в 1988 году и нацеленная на эффективное распознавание образов, входит в состав технологий глубокого обучения.

В обычном перцептроне каждый нейрон связан со всеми нейронами предыдущего слоя, причем каждая связь имеет отдельный вес. Сверточная нейронная сеть использует ограниченную **матрицу** весов небольшого размера, которую передвигают по всему слою. Такую матрицу весов называют **ядром свертки** или **фильтром**.

**Вектор** – одномерный массив. **Матрица** – двумерный массив.

Сверточная нейронная сеть использует операцию свертки или слой свертки.

**Как работает операция свертки?**

Чтобы лучше понять операцию свертки представим входные данные в виде матрицы (картинки).

Фильтр двигается по всем входным матрицам (**картам**) с определённым шагом. На каждом шаге значения фильтра перемножаются с значениями части карты, на которой находится фильтр и складываются. Получившееся значение записывается на новую карту, которая передается на следующий слой.

**Смысл перемножения карты и фильтра?**

Если фильтры будут правильно обучены, то мы на выходе получим набор признаков определенного объекта.

Можно сказать, что мы получим картинку, где будут высвечены черты нужного объекта. С помощью операции свертки нейросети будет проще предположить наличие на изображении нужного объекта.

**Подвыборочный слой (пулинг-слой)**

Чтобы нейросеть работала правильно обычно используют несколько слоёв свертки, в которых немало фильтров (от 30 и более), поэтому на выходе мы получим много данных, которые сильно нагружают компьютер. **Подвыборочный слой** нацелен на уменьшение объемов данных.

**Max Pooling**

Max Pooling – один из видов операции подвыборки. Он заключается в том, что карта (или изображение) разделяется на части, и в каждой части выделяется самое большое значение, а все остальные удаляются. Таким образом мы можем уменьшить карты признаков, сохранив сами признаки. Или можно уменьшить разрешение картинки, сохранив на ней все детали.

**Полносвязный слой**

Последний из типов слоев – это слой обычного перцептрона. Его цель – классификация. На вход он получает сконвертированные карты признаков, а на выходе выдаёт вероятности того, что определённые образы находятся на изображении.

**Как создать свою нейросеть и как её обучить?**

Создание своей нейросети включает несколько этапов.

1. **Подготовка данных**. Соберите и подготовьте набор данный (**датасет**), который будет использоваться для обучения.
1. **Определение архитектуры нейросети**. Определите количество слоев, типы слоёв, параметры слоев (количество нейронов, размеры фильтров, **функции активации**).
1. **Реализация нейросети**. Используйте фреймворк по выбору (**TensorFlow**, **PyTorch**, **Keras** и другие)
1. **Обучение нейросети**. Для обучения нейросетей подберите соответствующее оборудование, которое сможет справиться с требуемыми объемами данных, так как процесс обучения сильно зависит от характеристик компьютера. Перед обучением настройте параметры обучения (определение функции потерь, алгоритм оптимизации и другое), а также настройте гиперпараметры (скорость обучения, количество эпох обучения, размер пакета и другое). Сам процесс обучения занимает немало времени. Многие современные нейросети могут обучаться несколько часов или даже дней, но простые нейросети на обучение могут использовать всего несколько минут. Время зависит от размеров данных и от мощности оборудования.
1. **Тестирование нейросети**. Протестируйте нейросеть на тестовом наборе. Определите точность и среднюю абсолютную ошибку нейросети. Если результаты неудовлетворительны, то можно улучшить нейросеть: продолжить обучение (может нейросети нужно больше времени на обучение), поменять параметры сети или пересмотреть архитектуру сети и обучить нейросеть заново. Для тестирования используйте данные, которые не предоставлялись во время обучения.
1. Теперь обученную нейросеть можно использовать в тех задачах, для которых предназначалась нейросеть.

**Как я создавал собственную нейросеть?**

Для меня тема машинного обучения и компьютерного зрения очень заинтересовала, поэтому я захотел попробовать самостоятельно обучить нейросеть на простые задачи. Например, классификация изображений с собаками и кошками.

На данный момент самым популярным языком программирования для машинного обучения является **Python**. У Python существует много удобных библиотек, которые позволяют создавать нейронные сети, а также этот язык очень удобен для работы с базами данных.

**TensorFlow** – открытая программная библиотека для машинного обучения, разработанная компанией **Google** для решения задач построения и тренировки нейронной сети с целью автоматического нахождения и классификации образов, достигая качества человеческого восприятия.
```
import tensorflow as tf
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
```

Прежде чем создавать архитектуру модели, я собрал базу данных для обучения.

В интернете на данный момент существует уже много готовых баз данных на любые темы, которые доступны любому человеку, поэтому я смог собрать базу данных из 24 000 фотографий собак и кошек. Эти фотографии будут данными для обучения и тестирования.

Прежде чем отправить эту базу данных на обучение, ее надо отфильтровать от повреждённых файлов.

```
import os
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)
print("Deleted %d images" % num_skipped)
```

Загружаю базу данных в программу и привожу их в нужный вид

```
image_size = (180, 180)
batch_size = 128
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
```

Следующая задача создание архитектуры модели нейросети и настройка ее параметров.

```
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2)
])
```

Далее я настраиваю само обучение

```
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
```

Обучаю нейросеть, используя 25 **эпох** (При обучении нейросеть 25 раз пройдется по всем картинкам базы данных)

```
epochs = 25
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}"),
]
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
```

Тестирую нейросеть

```
model.evaluate(x_test,  y_test, verbose=2)
```

Сохраняю нейросеть

```
model.save('modal_cat_and_dog')
```
