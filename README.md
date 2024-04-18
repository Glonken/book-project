# Описание
Проект для подготовки пользовательских рекомендаций по прочтению книг с использованием технологий NLP.
Для демонстрации подготовлено отдельное веб-приложение.

# Подготовка

## Модель

```bash
user@host:~/git/book-project$ pip3 install -r reqs/model-requirements.txt
```

## Веб-приложение

```bash
user@host:~/git/book-project$ pip3 install -r reqs/web-requirements.txt
```

# Создание модели

```bash
user@host:~/git/book-project$ python3 create_model.py
```

# Запуск веб-приложения

```bash
user@host:~/git/book-project$ streamlit run web-app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.17.26.17:8501
```
