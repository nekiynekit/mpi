## MPI программы для зачёта по высокопроизводительным вычислениям
### Структура репозитория
`sources/task_i.py` - выполненное задание под номером `i`.

`utils/create_data_for_task_i.py` - скрипт подготовки данных для задания `i`.

`utils/calc_stats` - скрипты для генерации графиков в README.md
### Запуск
Все скрипты (кроме подготавливающих графики) написаны с использованием CLI библиотеки `click`, данные о запуске можно получить выполнив команду 
```bash
python sources/<имя скрипта>.py --help
```
### Графики 
![alt text](https://github.com/nekiynekit/mpi/plots/blob/main/result_1.png)
