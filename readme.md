# **Задача генеративной суммаризации** *(Суммаризация комментариев в социальных медиа)*

# **The task of generative summarization** *(Summarizing comments on social media)*

***Данная работа является решением реальной задачи онлайн-контеста [**Brand Analytics ML Contest**](https://ba-contest.ru/)., проходившего в декабре 2023 г.***

## Оглавление

1. [Описание работы](#описание-работы)

2. [Описание данных](#описание-данных)

3. [Зависимости](#зависимости)

4. [Установка и использование](#установка-и-использование)

5. [Описание решения](#описание-решения)

6. [Авторы](#авторы)

7. [Результат](#результат)

## Описание работы

**Описание задачи:**

В социальных медиа (vk, ok и т.п.) пользователи как правило могут оставлять посты, посты в группе, сообществе или на своей странице. Посты можно комментировать, однако комментарии при этом могут как иметь отношение к тексту поста, так и нет. Комментарий может содержать смысл или быть бессмыслицей или спамом в целом.

Необходимо реализовать решение, которое сможет генерировать (генеративная суммаризация) текст суммаризации (главного, смысла обсуждения) комментариев под каждым постом в нескольких режимах (типах).

**Правила суммаризации по уровню сложности:**

1. all_comments: суммаризация всех комментариев под каждым постом, без анализа самого поста;
2. post_comments: суммаризация только тех комментариев, которые имеют явное отношение к тексту каждого поста;
3. topic_comments: суммаризация комментариев которые имеют косвенное отношение к посту (пример: пост про технологию компании, а комментарий про обсуждение самой компании)

**Требования к решению:**

- использование только открытых технологий;
- запрещено использование в конечном результате (но допускается в процессе разработки) облачных решений: OpenAI и т.п.;
- конечное решение должно иметь инструкцию по запуску и установке всех зависимостей. Все внешние файлы, словари, модели и т.п. должны предоставляться вместе с самим решением;
- приложение должно иметь одну точку входа и формат вывода результата;
- ограничений по стеку технологий нет, но предпочтителен стандартный набор современного DS/ML: Python;
- высокая скорость работы: до 2 секунд на 10 комментариев для одного поста,

*Важно: Качество решения -- Ресурсоэффективность (чем меньше потребляется ресурсов — тем лучше) -- Скорость работы -- Предпочтительны решения, способные эффективно работать на CPU, допускаются решения с работой на GPU.*

[Оглавление](#оглавление)

**Структура работы:**

[data](./data). - образец файла с данными и три файла с вариантами сумаризации
   - [example.jsonl](./example.jsonl). - пример файла с данными (1 пост и 19 комментариев)
   - [result_all_comments.jsonl](./example.jsonl). - пример файла с результатом для типа: все комментарии (15 комментариев)
   - [result_post_comments.jsonl](./example.jsonl). - пример файла с результатом для типа: комментарии к посту (2 комментария)
   - [result_topic_comments.jsonl](./example.jsonl). - пример файла с результатом для типа: комментарии к теме (13 комментариев)

[src](./src). - исходные файлы для сборки
   - [moduls.py](moduls.py). - необходимые функции, импорты библиотек и модулей
   - [stopwords-en.txt](stopwords-en.txt). - английские стоп-слова
   - [stopwords-ru.txt](stopwords-ru.txt). - русские стоп-слова

[solution.py](./solution.py). - исполняемый файл решения

[dependencies.txt](./dependencies.txt). - зависимости (при необходимости)

**Метрики:**

Для каждого типа суммаризации есть проверочные эталонные корпуса, которые сравниваются с итоговой суммаризацией решения.

***Критерии оценки:***

- Точность по метрикам bert score и rouge
- Полнота по метрикам bert score и rouge
- F1 по метрикам bert score и rouge
- Производительность (скорость работы)
- Потребление ЦПУ
- Потребление ГПУ
- Потребление ОЗУ

*Дополнительно: предпочтительны решения, способные эффективно работать на CPU.*

[Оглавление](#оглавление)

## Описание данных

Файл в формате .jsonl (1 jsоn-объект на 1 строку) с постами (в т.ч. ссылками на видео с YouTube и т.п.) и комментариями из VK, Telegram и YouTube.

Данные в файле представлены в хаотичном порядке, т.е. необходимо связать комментарии и посты по внешним идентификаторам (root_id и id, соответственно), которые указаны в качестве отдельного поля каждого объекта исходного файла, а также провести базовые операции (очистка и т.п.) предобработки.

**Пример данных:**

***// Пост***

{

    "text": "string",

    "url": "http://www.youtube.com/...",

    "id": "_UYt9yJFck0",

    "hash": "008c23470d1454e362218325921370f1",

    "date": 1699145999

}

***// Комментарий***

{

    "text": "string",

    "url": "http://www.youtube.com/...",

    "id": "-354t9yJFck0",

    "hash": "008c23470d1454e362218325921370f1",

    "root_id": "_UYt9yJFck0",

    "parent_id": "-2353439yJFck0",

    "date": 1700147948

}

[Оглавление](#оглавление)

## Зависимости

- jsonlines==4.0.0
- natasha==1.6.0
- navec==0.10.0
- numpy==1.23.5
- pandas==2.1.0
- scikit-learn==1.3.0
- transformers==4.36.2
- wget==3.2

## Установка и использование

Нажать зеленую кнопку: <> Code, далее выбрать нужное (клонировать репозиторий, открыть на ПК или скачать его архив), при запуске кода необходимо учесть существующие зависимости.

Запустить файл solution.py, передав в него (например, в командной строке) три агрумента: тип суммаризации ( all_comments, post_comments или topic_comments), путь к файлу с данными (напрмер имеющийся example.jsonl) и путь и назвение итогового файла. Пример:

**./solution all_comments ./example.jsonl ./result.jsonl**

По итогу выполнения любого из типов суммаризации будет сформирован .jsonl файл со следующей структурой:

{

"summary": "string", // текст суммаризации

"post_hash": [], // hash исходного поста

"comments_hash": [], // hash комментариев, подошедших к суммаризации

}

*Дополнительно: после первого запуска в папку /src будет загружен файл navec_hudlit_v1_12B_500K_300d_100q.tar  - коллекция предобученных эмбеддингов для русского языка (размер ~ 50 МБ). Т.е. этот объем должен быть скачан из репозитория.*

[Оглавление](#оглавление)

## Описание решения

Общая задача подразделяется на две подзадачи:
1. Выбор комментариев исходя из типа суммаризации (все комментарии, только относящиеся к посту или относящиеся к теме поста, т.е. предполагается, что комментарии первого типа это сумма комментариев второго и третьего типов)
2. Генеративная суммаризация отобранных комментариев

Дополнительно необходимо решить сопутствующие подзадачи:
- удаление абсолютного спама (т.е. комментарии, не имеющие ни какого смысла, рекламные, ошибочные и т.д.), т.е. вообще удалить такие комментарии из рассмотрения
- очистка текста как поста, так и комментариевю причем очистка должна быть двустадийная: очистка перед эмбеддингом текста и дополнительная очистка перед передачей текста в модель (исходя из того, что будут использоваться два разных эмбеддинга для выбора комментраиев и суммаризации текстов)

Тексты очищаются дважды (следует учесть, что тексты комментариев в соцсетях очень "грязные" и "мусорные" во всех смыслах):
- для векторизации с целью очистки от спама и отбора комментариев: остаются русские и латинские символы (русский и английский текст), точки, заглавные буквы и некоторые символы.
- для модели с целью суммаризации тексты дополнительно очищаются от стоп-слов, добавляется точка в конце текста, при отсутствии, удаляются цифры и латинские символы (остается только русский текст) и  т.д.

Для очистки от спама, выбора комментариев используются эмбеддинги Navec (библиотека Natasha) (GloVe-эмбеддинги обученные на 145ГБ художественной литературы). Для каждого слова текста комментария берется эмбеддинг из словаря Navec (вектор размерности (300, )). Из получившейся матрицы 300 х N (N - количество слов, имеющих эмбеддинг в Navec) вычисляется вектор документа комментария (вектор со средними соответствующих элементов векторов слов). Для всех векторов документов комментариев определяется средний вектор документов комментариев (аналогично). Далее спам определяется как комментарии, чьи вектора документа комментария имеют косинусную близость со средним ветором документов комментариев менее 0.4. Такиие комментарии отбрасываются.

Для разделения комментариев по типам также используются эмбеддинги слов Navec. Только вектор документа комментария вычисляется по другому принципу - для получившейся матрицы 300 х N осуществляется понижение размерности с N до 1 и получается вектор размерности (300, ), т.е. размерности 1 вместо N. В качестве методов понижения размерности используются 5 разных способов: PCA, KernelPCA, TruncatedSVD, Isomap и SpectralEmbedding и вычисляется их среднее (вектор средних значений). Далее аналогично определяетя вектор документа поста. Если вектор документа поста не существует (нет текста до или после очистки или нет эмбеддинга ни для одного слова) или если требуется тип суммаризаци - все комментарии, то все  имеющиеся комментарии считаются подходящими и используются для суммаризации. Если вектор документа поста существует, то комментарии разделяются с помощью кластеризации (векторов документов комментариев) на 2 кластера (метод K-means). Вычисляется косинусная близость центров кластеров к вектору документа поста и кластер с более близким центром считается содержащим вектора документов комментариев к посту, а с более дальним центром - содержащим вектора документов комментариев к теме поста. Учтен случай только одного комментария.

После определения "подходящих" комментариев их список передается на суммаризацию.

Все отобранные комментарии делятся на группы, по степени схожести. Для разделения используется кластеризация, а именно метод K-means, с определением оптимального количества кластеров методом коэффициента силуэта. При количестве комментариев от 1 до 5, они передаются в модель суммаризации все, по очереди. Свыше 5 - кластеризация. Далее для каждого кластера выбираются не более пяти комментариев, вектора документов которых максимально близки по косинусной близости к центру своего кластера и их объединенный текст передается в модель суммаризации. При этом в зависимости от количества комментариев (от 1 до 5) передается параметр: минимальная длина, используемая по умолчанию в методе генерации модели суммаризации.

В качестве модели суммаризации используется [rut5-base-absum](https://huggingface.co/cointegrated/rut5-base-absum). Это обрезанная модель mT5 (оставлены русский и английский языки) и дообученная на корпусе из 4 источников.

Параметры модели для инференса подобраны экспериментально, вручную (суммаризацией различных текстов).

Следует отметить, что на всех этапах отслеживается является ли пустым текст (отсутствие символов или наличие единственного пробела) и существуют ли эмбеддинги (соответствующие вектора).

[Оглавление](#оглавление)

## Авторы

[Артем Корнев](https://t.me/@ArtemKornev0)

## Результат

Данная работа является решением реальной задачи онлайн-контеста [**Brand Analytics ML Contest**](https://ba-contest.ru/)., проходившего в декабре 2023 г.

В мероприятии зарегистрировались 167 участников (команд и отдельных лиц). Итоговые решения подали 14 участников. Данное решение заняло **7 место** набрав 140 баллов (14 поданных решений набрали от 25 до 214 баллов).

[Оглавление](#оглавление)
