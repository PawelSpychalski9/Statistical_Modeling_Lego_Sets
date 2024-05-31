import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv('lego_sets.csv')

#zapoznanie się z danymi
def data_check (lego_sets):
    print("Tabela")
    print(lego_sets.head())
    print(10*"-")
    print("Opis")
    print(lego_sets.describe())
    print(10*"-")
    print("Informacje")
    print(lego_sets.info())
    print(10*"-")
    print("Braki danych")
    print(lego_sets.isnull().sum())
    print(10*"-")
    print("Kolumny")
    print(lego_sets.columns)
    print(10*"-")
    print("Typy danych")
    print(lego_sets.dtypes)
    print(10*"-")
    print("Liczba wierszy")
    print(len(lego_sets["ages"]))
    print(10*"-")

#zadanie 4
def calculate_statistics(df, column_name):
    mean = round(df[column_name].mean(), 2)
    median = round(df[column_name].median(), 2)
    std = round(df[column_name].std(), 2)
    var = round(df[column_name].var(), 2)
    skewness = round(df[column_name].skew(), 2)
    kuri = round(df[column_name].kurtosis(), 2)
    _3sigma = round(np.sum((df[column_name] > mean - 3 * std) & (df[column_name] < mean + 3 * std)) / len(df) * 100, 2)

    if skewness > 0:
        skewness_description = "prawostronna"
    elif skewness < 0:
        skewness_description = "lewostronna"
    else:
        skewness_description = "brak skośności"

    #odejmujemy 3, bo wartość kurtosis dla rozkładu normalnego wynosi 3
    kuri_excess = kuri - 3
    if kuri > 0:
        kuri_description = "rozkład leptykurtyczny"
    elif kuri < 0:
        kuri_description = "rozkład platykurtyczny"
    else:
        kuri_description = "rozkład mezokurtyczny"

    print(f"Średnia dla {column_name}: {mean}")
    print(f"Mediana dla {column_name}: {median}")
    print(f"Odchylenie standardowe dla {column_name}: {std}")
    print(f"Wariancja dla {column_name}: {var}")
    print(f"Skośność dla {column_name}: {skewness} ", f"\t | \t Opis: {skewness_description} skośność")
    print(f"Kurtoza dla {column_name}: {kuri}", f"\t | \t Opis: {kuri_description} ", f"\t | \t Nadmiar kurtozy: {kuri_excess}")
    print(f"Procent wartości mieszczących się w przedziale 3 sigma dla {column_name}")

#zadanie 5 i 6
def calculate_and_comapare(df, column_name, number_of_intervals):
    sorted_values = df[column_name].sort_values()
    min_values = sorted_values.min()
    max_values = sorted_values.max()

    print(f"Min wartość dla {column_name}: {min_values}")
    print(f"Max wartość dla {column_name}: {max_values}")

    #obliczenie szerokości przedziału
    interval_width = (max_values - min_values) / number_of_intervals

    # Utworzenie interwałów
    intervals = [min_values + i * interval_width for i in range(number_of_intervals + 1)]

    print(10 * "=")

    sorted_values = sorted_values.to_frame()

    # Przypisanie interwałów do danych za pomocą pd.cut
    sorted_values['interval'] = pd.cut(sorted_values[column_name], bins=intervals,
                                       labels=[f'interval_{i}' for i in range(number_of_intervals)])

    #sprawdzenie który przedział jest najliczniejszy
    print(sorted_values["interval"].value_counts())
    most_frequent_interval = sorted_values["interval"].value_counts().idxmax()
    TEMP_df = sorted_values[sorted_values["interval"] == most_frequent_interval]

    #print(TEMP_df)
    print(10 * "=")
    print(f"Wartosci statystyczne dla {number_of_intervals} przedziałów dla {column_name}")
    calculate_statistics(TEMP_df, column_name)
    print(10 * "=")

#zadanie 7
def goodness_of_fit(df, column_name, alpha):
    # Wyodrębnienie danych z kolumny
    data = df[column_name].values

    # Testowanie zgodności z rozkładem normalnym
    print("Test zgodności z rozkładem normalnym:")
    normality_test = stats.normaltest(data)
    print("Statystyka testowa:", normality_test.statistic)
    print("P-value:", normality_test.pvalue)
    if normality_test.pvalue < alpha:
        print(f"Dla poziomu istotności {alpha}, odrzucamy hipotezę zerową o normalności rozkładu.")
    else:
        print(f"Dla poziomu istotności {alpha}, nie ma podstaw do odrzucenia hipotezy zerowej o normalności rozkładu.")

    # Testowanie zgodności za pomocą testu Kołmogorowa-Smirnowa
    print("\nTest Kołmogorowa-Smirnowa:")
    ks_test = stats.kstest(data, 'norm')
    print("Statystyka testowa:", ks_test.statistic)
    print("P-value:", ks_test.pvalue)
    if ks_test.pvalue < alpha:
        print(f"Dla poziomu istotności {alpha}, odrzucamy hipotezę zerową o zgodności rozkładu z rozkładem normalnym.")
    else:
        print(f"Dla poziomu istotności {alpha}, nie ma podstaw do odrzucenia hipotezy zerowej o zgodności rozkładu z rozkładem normalnym.")

    # Testowanie zgodności za pomocą testu chi-kwadrat
    print("\nTest chi-kwadrat:")
    chi2, p_value = stats.chisquare(data)
    print("Statystyka testowa:", chi2)
    print("P-value:", p_value)
    if p_value < alpha:
        print(f"Dla poziomu istotności {alpha}, odrzucamy hipotezę zerową o zgodności rozkładu.")
    else:
        print(f"Dla poziomu istotności {alpha}, nie ma podstaw do odrzucenia hipotezy zerowej o zgodności rozkładu.")

#sprawdzenie danych
data_check(df)

#nazwy cech
attribute1 = "list_price"
attribute2 = "piece_count"


#liczba przedziałów do obliczenia statystyk
n = 10

#wartość alfa
alfa = 0.05

#wywołanie funkcji dla obu cech
calculate_statistics(df, attribute1)
calculate_statistics(df, attribute2)
calculate_and_comapare(df, attribute1, n)
calculate_and_comapare(df, attribute2, n)
goodness_of_fit(df, attribute1, alfa)
goodness_of_fit(df, attribute2, alfa)