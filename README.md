# Koło czy kwadrat

Program uczy się odróżniać koło od kwadratu i rozpoznaje figury narysowane myszką.

Program składa się z **trzech oddzielnych, samodzielnych programów**. Każdy uruchamia się osobno
(`python nazwa.py`) i ma własne okno. Łączą je tylko pliki z danymi: `data.csv` i `boundary.json`.

Wymagane biblioteki: numpy, Pillow, matplotlib (tkinter jest w standardowym Pythonie).

## Kolejność pracy

| Krok | Program | Do czego służy |
|---|---|---|
| 1 | `python train.py` | **Uczenie.** Komputer sam rysuje tysiące kół i kwadratów („Generuj", dopisuje do `data.csv`) albo rysujesz figurę i oznaczasz ją „To koło" / „To kwadrat". Po każdym dopisaniu model uczy się na nowych figurach. |
| 2 | `python recognize.py` | **Rozpoznawanie.** Rysujesz figurę jednym pociągnięciem (obrys sam się domyka). Program mówi „koło" albo „kwadrat" i pokazuje na wykresie 2D, gdzie figura wypadła względem danych i granicy. |
| 3 | `python simulate.py` | **Symulacja.** Animacja uczenia na wykresie 2D: punkty pojawiają się wiersz po wierszu, a granica się przesuwa. Niczego nie uczy, tylko odtwarza `weights.csv`.

Sterowanie symulacją:
- **Prędkość:** suwak od 0 (animacja stoi) w górę, w skali logarytmicznej. Animacja idzie tylko do przodu.
- **Krok:** suwak pozycji i pole z numerem kroku. Przeskakują do dowolnego kroku, także wstecz, a animacja rusza stamtąd do przodu. Numer mniejszy niż 1 daje krok 1, większy niż ostatni daje ostatni krok.
- **Start (od początku)** zaczyna od kroku 1, a **Stop** zatrzymuje animację. |

Można też uczyć bez okna: `python model.py` (jedno przejście po całym `data.csv`).

## Wykres 2D

Oś X to **kolistość**, oś Y to **liczba rogów**. Koła są niebieskie, kwadraty czerwone. Czarna przerywana
linia to granica decyzji (z = 0), a tło pokazuje, po której jej stronie model mówi „koło", a po której
„kwadrat". Liczba rogów jest całkowita, więc punkty mają na wykresie drobne przesunięcie w pionie,
żeby się nie zasłaniały (same dane nie są zmieniane).

## Jak model uczy się i gdzie zapisuje wagi

Model uczy się **po jednej figurze**. Jest jeden, ciągły trening: każde kolejne uczenie zaczyna od wag,
na których skończyło poprzednie. Wagi (trzy liczby: `w_circularity`, `w_corners`, `bias`) trafiają do dwóch plików:

- `weights.csv`: **cała historia**. Po każdym kroku dopisywany jest jeden wiersz, nigdy nic się nie kasuje. Odtwarza go `simulate.py`.
- `boundary.json`: **ostatnie wagi**, nadpisywane po każdym uczeniu. Używa ich `recognize.py`.

Po dopisaniu figur `train.py` uczy się tylko na nowych. Przycisk „Ucz ponownie na całym data.csv” robi jeszcze jedno przejście po wszystkich figurach.

Wartości średnie i skale cech są stałymi w kodzie (`model.py`), a nie liczone z danych.

Trafność w oknie uczenia liczona jest uczciwie: każda figura jest oceniana wagami z chwili **przed** nauką na niej.

## Co robią poszczególne pliki

| Plik | Co robi |
|---|---|
| `train.py` | Program 1: okno uczenia. |
| `recognize.py` | Program 2: okno rozpoznawania. |
| `simulate.py` | Program 3: okno symulacji. |
| `model.py` | Model (regresja logistyczna na 2 cechach, nauka figura po figurze), zapis `weights.csv` i `boundary.json`. |
| `chart.py` | Wspólny wykres 2D (osie, tytuł, legenda, granica) dla programów 2 i 3. |
| `drawing_canvas.py` | Płótno do rysowania jednym pociągnięciem (programy 1 i 2). |
| `feature_extractor.py` | Zamienia narysowaną linię na 2 liczby (cechy): kolistość i liczbę rogów. |
| `data_generator.py` | Komputer „rysuje" w pamięci koła i kwadraty (z drżeniem ręki, owalnością, niedomkniętą linią) i liczy ich cechy. |
| `data_handler.py` | Odczyt i zapis `data.csv`. |

Pliki z danymi:

| Plik | Zawartość |
|---|---|
| `data.csv` | Jeden wiersz to jedna figura: kolistość, liczba rogów, etykieta, źródło. Figury tylko się dopisuje. |
| `weights.csv` | Historia wag: trzy kolumny, po jednym wierszu na każdy krok uczenia. Tylko dopisywany. |
| `boundary.json` | Ostatnie wagi, których używa `recognize.py`. |
| `simulation.png` | Ostatnia klatka symulacji. |

Przepływ danych:

```
generator (lub Twój rysunek) -> cechy -> data.csv -> train.py -> weights.csv (historia) -> simulate.py
                                                        \-> boundary.json (ostatnie wagi) -> recognize.py
```

## Na czym polegają cechy

Program nie „widzi" koła ani kwadratu. Z narysowanej linii wylicza **2 liczby** opisujące kształt, a decyzję podejmuje dopiero na ich podstawie. Obie są względne, więc nie zależą od wielkości ani obrotu figury.

### 1. Kolistość

Wyobraź sobie gumkę naciągniętą wokół figury. Program porównuje pole tej gumki z jej obwodem.

- **Idealne koło** daje ok. 1,0.
- **Kwadrat** daje ok. 0,78–0,88 (narysowany odręcznie).
- **Owal** wychodzi pośrodku.

Gumka „nie widzi" wgłębień, więc drżenie ręki prawie nie zmienia tej cechy.

### 2. Liczba rogów

Program upraszcza tę gumkę do wielokąta o jak najmniejszej liczbie wierzchołków, tak żeby kształt prawie się nie zmienił, i liczy wierzchołki.

- **Kwadrat:** 4 (odręcznie 4–5).
- **Koło:** ok. 8. To nie są prawdziwe rogi, tylko tyle odcinków potrzeba, żeby przybliżyć okrąg przy tej tolerancji.

Zaokrąglone rogi kwadratu zwiększają tę liczbę, więc cecha jest pewna dla ostrych kwadratów, a mniej dla bardzo zaokrąglonych.

### Typowe wartości (z 20 narysowanych odręcznie figur)

| Cecha | Koła | Kwadraty |
|---|---|---|
| Kolistość | 0,91–0,99 | 0,79–0,88 |
| Liczba rogów | 7–8 | 4–5 |

## Jak model łączy dwie cechy

Model używa dwóch cech: kolistości i liczby rogów. Każda ma wagę. Model sumuje
*waga × cecha* dla obu cech, plus stałą. Wynik z dodatni oznacza kwadrat, ujemny lub zero koło.
Zbiór punktów, gdzie z = 0, to na wykresie 2D prosta, czyli granica.
