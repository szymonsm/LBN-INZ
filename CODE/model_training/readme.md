Struktura plików w MODEL_TRAINING:

    - foldery BA, BTC, NFLX, TSLA:
        - 11_features.ipynb - notebook z preprocessingiem danych dla danego ticker'a,
                            łączenie z danymi z newsów, usuwanie braków danych, usuwanie weekendów
        
        - 22_selection.ipynb - notebook z wyborem cech dla danego ticker'a
                               na podstawie wielkorotnego trenowania modeli i analizy SHAP

        - 33_lstm.ipynb - notebook z trenowaniem modeli z warstwami LSTM dla danego ticker'a
                               
        
    - folder csv - zawiera utworzone pliki csv z danymi dla danego ticker'a

    - folder models - zawiera utworzone modele dla danego ticker'a

    - folder scalers - zawiera utworzone skalery dla danego ticker'a

    - folder scripts - zawiera skrypty pomocnicze

    - folder horizon_results - zawiera wyniki z modelowania horyzontalnego modeli NBEATSx, NHITS, TFT

    - eda.ipynb - notebook ze wstępną analizą danych i utworzeniem wykresów do pracy

    - results_lstm.ipynb - notebook z testowaniem strategii inwestycyjnej i końcowym podsumowaniem wyników dla modeli z LSTM

    - horizon_train_main.ipynb - notebook służący do modelowania horyzontalnego

    - horizon_results_main.ipynb - notebook do policzenia wyników oraz strategii
