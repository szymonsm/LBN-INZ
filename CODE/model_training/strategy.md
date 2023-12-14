STRUKTURA

+ notebook 1 
- odpalić dane jakieś wykresy pogląd ostatni rok
- wyliczenie ciekawostek jaka zmiana na dzień np
- łączyć z newsami
- dodać zero gdzie braki
- dodać mean po modelach jako nowe kolumny
1. zapis
- uzupełnić jako bez weekendu
- dodać nowe wyliczane zmienne
2. zapis
- feedforward
- dodać nowe wyliczane kolumny

notebook 2
- dodać zmienną celu czyli nastepny tydzień itp jako funkcja
- wybór jakie zmienne mniej więcej użyć
- najwyżej mocny split zrobić żeby cały 2023 zostawić

notebook 3
- ~~podział na kilka zbiorów - timeseries split 1msc val 2msc test x3~~
- model baseline który daje następny dzień jako predykcję
- podzbiory kolumn
- użycie propheta + tłumaczenie dlaczego na 2 zapisie - jak użyliśmy na zerach to słabo wyszło na poniedziałkowej predykcji
- wzięcie jego predykcji i zapis
- funkcje do fancy metryk jako df dla 3 okresów train test val

notebook 4
- split 
- dodanie kolumn z propheta
- zrobić min max na kolumnach / zapytać czata czy ma inny pomysł bo są blisko
- zapis do uczenia

notebook 5 
LSTM
- wybór okna czasowego
- podzbiory kolumn
- optuna na strukturze

- douczenie najlepszych opcji na colabie zapis modelu
- sprawdzenie funkcji do metryk

notebook 6
LGBM
- wybór okna czasowego
- podzbiory kolumn
- optuna na strukturze
- sprawdzenie funkcji do metryk

notebook 7 
NBEATS
- wybór okna czasowego
- podzbiory kolumn
- optuna na strukturze
- sprawdzenie funkcji do metryk

**dla tygodnia i miesiąca zamiast nbeats - może być PatchTST**

notebook 8 
NHEATS
- wybór okna czasowego
- podzbiory kolumn
- optuna na strukturze
- sprawdzenie funkcji do metryk
